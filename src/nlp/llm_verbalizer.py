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
    """
    if not _llm_ready or _model is None or _tokenizer is None:
        return None

    try:
        import torch
        import re

        # Determine sentiment from rating_skew in the prompt
        label = "positive"
        if "averaged 1" in prompt or "averaged 2" in prompt:
            label = "negative"
        elif "averaged 3" in prompt:
            label = "neutral"

        # Extract item name if present
        item_match = re.search(r"Item:\s*(.+)", prompt)
        item_name = item_match.group(1).strip() if item_match else "product"

        # Force the model into Pidgin mode by starting the review with "na" (is)
        # and using the exact format it was trained on.
        simple_prompt = f"### Review ({label}):\nThis {item_name} na"

        inputs = _tokenizer(simple_prompt, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,  # Slightly higher temp
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.25,  # Higher penalty to stop repeating "Geilege"
                pad_token_id=_tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        review = _tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Clean up any trailing markers
        review = re.sub(r"###.*", "", review, flags=re.IGNORECASE).strip()
        review = re.sub(r"RATING:.*", "", review, flags=re.IGNORECASE).strip()

        # Re-attach the start of the sentence
        full_review = f"This {item_name} na {review}"

        if full_review:
            logger.info(f"Generated review ({len(full_review)} chars): {full_review[:80]}...")
        return full_review if len(full_review) > 10 else None

    except Exception as e:
        logger.error(f"LLM generation failed: {type(e).__name__}: {e}")
        return None
