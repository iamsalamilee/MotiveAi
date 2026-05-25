"""
LLM Verbalizer — Loads the merged Qwen 2.5 1.5B model directly inside the Space.

The model at Iamsalamilee/motiveai-pidgin is already merged (base + LoRA fused),
so we load it as a complete model. Runs on CPU (slow but works on free tier).
"""
import os
import re
from loguru import logger

# ── Configuration ────────────────────────────────────────────────────────
MERGED_MODEL_ID = "Iamsalamilee/motiveai-pidgin"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_model = None
_tokenizer = None
_llm_ready = False
_check_count = 0


def load_llm() -> bool:
    """
    Load the merged Qwen 2.5 1.5B model directly from HuggingFace Hub.
    Runs on CPU in float32 (no GPU needed).
    """
    global _model, _tokenizer, _llm_ready, _check_count

    if _llm_ready:
        return True

    _check_count += 1
    if _check_count > 2:
        return False

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading merged model from {MERGED_MODEL_ID} (CPU, float32)...")

        _tokenizer = AutoTokenizer.from_pretrained(
            MERGED_MODEL_ID,
            token=HF_TOKEN if HF_TOKEN else None,
            trust_remote_code=True,
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        logger.info("Tokenizer loaded.")

        _model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_ID,
            token=HF_TOKEN if HF_TOKEN else None,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _model.eval()

        _llm_ready = True
        logger.info(f"Model loaded! Parameters: {sum(p.numel() for p in _model.parameters()):,}")
        return True

    except Exception as e:
        logger.warning(f"Model load failed: {type(e).__name__}: {e}")
        return False


def generate_review(prompt: str, max_new_tokens: int = 150) -> str:
    """
    Generate a review using the locally loaded model.
    Uses Chat Template and proper stop tokens to prevent hallucinations.
    """
    if not _llm_ready or _model is None or _tokenizer is None:
        return None

    try:
        import torch
        import re

        # Extract item name
        item_match = re.search(r"Item:\s*(.+)", prompt)
        item_name = item_match.group(1).strip() if item_match else "product"

        pidgin_words = ["dey", "sabi", "na", "sef", "sha", "abeg", "wahala", "them"]
        user_speaks_pidgin = any(word in prompt.lower() for word in pidgin_words)

        if user_speaks_pidgin:
            sys_msg = "You are a Nigerian product reviewer. You MUST write in Nigerian Pidgin English. Use words like 'e dey work', 'wahala', 'na', 'abeg'. Keep it short (2-3 sentences)."
        else:
            sys_msg = "You are a product reviewer. Write a standard English review. Keep it short (2-3 sentences)."

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Write a review for this product: {item_name}"}
        ]

        chat_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=512)

        # Qwen chat models use <|im_end|> (151645) as the stop token. 
        # If we don't tell it to stop there, it hallucinates indefinitely!
        stop_tokens = [_tokenizer.eos_token_id]
        if hasattr(_tokenizer, "convert_tokens_to_ids"):
            im_end = _tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None:
                stop_tokens.append(im_end)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=stop_tokens,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        review = _tokenizer.decode(generated, skip_special_tokens=True).strip()

        if review:
            logger.info(f"Generated review ({len(review)} chars): {review[:80]}...")
        return review if len(review) > 5 else None

    except Exception as e:
        logger.error(f"LLM generation failed: {type(e).__name__}: {e}")
        return None
