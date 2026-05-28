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
            torch_dtype=torch.bfloat16,
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
    Uses ultra-simple raw completion since the model hallucinates on chat templates.
    """
    if not _llm_ready or _model is None or _tokenizer is None:
        return None

    try:
        import torch
        import re

        # Determine sentiment
        label = "positive"
        if "averaged 1" in prompt or "averaged 2" in prompt:
            label = "negative"
        elif "averaged 3" in prompt:
            label = "neutral"

        # Extract item name
        item_match = re.search(r"Item:\s*(.+)", prompt)
        item_name = item_match.group(1).strip() if item_match else "product"

        # Only use words that are EXCLUSIVELY Pidgin (not common English words)
        # Use word boundaries (\b) so "sha" doesn't match "shall" or "shape"
        pidgin_only = ["dey", "sabi", "sef", "abeg", "wahala", "wetin", "oga"]
        user_speaks_pidgin = any(re.search(rf'\b{w}\b', prompt.lower()) for w in pidgin_only)

        # Force the model by starting the sentence for it.
        # This completely bypasses the need for complex instructions.
        if user_speaks_pidgin:
            if label == "positive":
                prefix = f"Review:\nMake I no lie, this {item_name} na proper correct thing. E dey"
            elif label == "negative":
                prefix = f"Review:\nAbeg, this {item_name} na pure wahala. E no dey"
            else:
                prefix = f"Review:\nThis {item_name} just dey there. E no too"
        else:
            if label == "positive":
                prefix = f"Review:\nI am very happy with this {item_name}. It is"
            elif label == "negative":
                prefix = f"Review:\nI am very disappointed with this {item_name}. It does not"
            else:
                prefix = f"Review:\nThis {item_name} is just okay. It"

        inputs = _tokenizer(prefix, return_tensors="pt", truncation=True, max_length=128)

        # Low temperature to prevent hallucinations ("Geilege", Chinese characters, etc)
        # No repetition penalty to avoid weird invented words like "ughtily"
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=30,  # Short and punchy
                do_sample=False,    # Greedy decoding = faster on CPU
                repetition_penalty=1.2,
                pad_token_id=_tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        review = _tokenizer.decode(generated, skip_special_tokens=True).strip()

        # ── AGGRESSIVE CLEANUP ──────────────────────────────────────────
        # The model was trained on sentiment classification data too,
        # so it sometimes leaks "What is the sentiment?" etc.

        # 1. Cut at the first newline — reviews don't have line breaks
        review = review.split("\n")[0].strip()

        # 2. Cut at any training artifact pattern
        artifact_patterns = [
            r"What is.*",        # "What is the sentiment..."
            r"Options are.*",    # "Options are: (1)..."
            r"Sta da.*",         # "Sta da best of luck..."
            r"The sentiment.*",  # "The sentiment of this review..."
            r"Choose.*",         # "Choose the correct..."
            r"Select.*",         # "Select the best..."
            r"Answer.*",         # "Answer:"
            r"Question.*",       # "Question:"
            r"\(1\).*",          # "(1)..."
            r"Review:.*",        # Another "Review:" starting
        ]
        for pattern in artifact_patterns:
            review = re.split(pattern, review, flags=re.IGNORECASE)[0].strip()

        # 3. Keep only the first complete sentence
        sentences = re.split(r'(?<=[.!?]) +', review)
        if len(sentences) > 1:
            review = sentences[0]

        # 4. Remove trailing markers
        review = re.sub(r"###.*", "", review, flags=re.IGNORECASE).strip()
        review = re.sub(r"RATING:.*", "", review, flags=re.IGNORECASE).strip()

        # 5. If it ends mid-sentence (no period), add one
        if review and review[-1] not in ".!?":
            review = review.rsplit(",", 1)[0] + "."

        # Attach the prefix back
        full_review = prefix.replace("Review:\n", "") + " " + review

        if full_review:
            logger.info(f"Generated review: {full_review[:120]}...")
        return full_review if len(full_review) > 10 else None

    except Exception as e:
        logger.error(f"LLM generation failed: {type(e).__name__}: {e}")
        return None
