"""
LLM Verbalizer — Loads Mistral 7B + LoRA adapter for Nigerian Pidgin review generation.

Uses 4-bit quantization (QLoRA) to fit the 7B model in ~4GB VRAM.
Gracefully falls back to a rule-based placeholder if:
  - No GPU is available (CPU-only deployment)
  - LoRA weights haven't been uploaded yet
  - Any loading error occurs
"""
from loguru import logger

_model = None
_tokenizer = None
_llm_ready = False

# ── Configuration ────────────────────────────────────────────────────────
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
LORA_MODEL_ID = "Iamsalamilee/motiveai-pidgin-lora"  # Change this to your actual HF model ID


def load_llm():
    """
    Attempt to load Mistral 7B + LoRA with 4-bit quantization.
    Returns True if successful, False if fallback is needed.
    """
    global _model, _tokenizer, _llm_ready

    if _llm_ready:
        return True

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("No GPU detected. LLM will use placeholder mode.")
            return False

        logger.info(f"Loading base model: {BASE_MODEL_ID} (4-bit quantized)...")

        # 4-bit quantization config — identical to training config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model with quantization
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA adapter on top
        logger.info(f"Loading LoRA adapter: {LORA_MODEL_ID}...")
        _model = PeftModel.from_pretrained(_model, LORA_MODEL_ID)
        _model.eval()

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _llm_ready = True
        logger.info("LLM loaded successfully! Nigerian Pidgin mode activated.")
        return True

    except Exception as e:
        logger.warning(f"LLM failed to load (using placeholder): {e}")
        return False


def generate_review(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Generate a review using the loaded LLM.
    Falls back to None if LLM is not loaded.
    """
    if not _llm_ready or _model is None or _tokenizer is None:
        return None

    try:
        import torch

        inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # Decode only the generated part (skip the prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        review = _tokenizer.decode(generated, skip_special_tokens=True).strip()

        return review

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return None
