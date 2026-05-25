"""
Task A — User Modeling & Review Simulation.

Full pipeline:
  1. Normalize review texts (Levenshtein spelling correction)
  2. Score sentiment (Pidgin VADER)
  3. Extract user profile (APG4RecSim)
  4. Embed normalized texts (NaijaBERT)
  5. Temporal attention weighting (SASRec)
  6. Build LLM prompt from profile + target item
  7. Generate review (placeholder until LoRA model arrives)
  8. Parse rating from output
"""
from pathlib import Path
from fastapi import APIRouter
from loguru import logger

from src.app.models.schemas import (
    SimulateReviewRequest,
    SimulateReviewResponse,
    ProfileTraits,
)

router = APIRouter()

# ── Lazy-loaded singletons (initialized at startup via main.py) ──────────
_profile_gen = None
_temporal = None
_naija_bert = None
_prompt_template = None


def _get_components():
    """Lazy-load all pipeline components on first request."""
    global _profile_gen, _temporal, _naija_bert, _prompt_template

    if _profile_gen is None:
        from src.agent.apg4recsim import ProfileGenerator
        _profile_gen = ProfileGenerator()
        logger.info("Loaded APG4RecSim ProfileGenerator")

    if _temporal is None:
        from src.agent.sasrec import temporal_attention
        _temporal = temporal_attention
        logger.info("Loaded SASRec TemporalAttention")

    if _naija_bert is None:
        from src.nlp.naija_bert import naija_bert
        _naija_bert = naija_bert
        logger.info("Loaded NaijaBERT embedder")

    if _prompt_template is None:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "simulate_review.txt"
        _prompt_template = prompt_path.read_text(encoding="utf-8")
        logger.info(f"Loaded prompt template from {prompt_path}")

    return _profile_gen, _temporal, _naija_bert, _prompt_template


# ── LLM Review Generator — tries real model, falls back to placeholder ────
def _generate_review(prompt: str, profile) -> tuple[str, float]:
    """
    Generate a review using the LoRA-finetuned LLM if available.
    Falls back to a rule-based placeholder if the LLM isn't loaded.

    Returns:
        (generated_review, predicted_rating)
    """
    from src.nlp.llm_verbalizer import load_llm, generate_review

    rating = round(profile.rating_skew)
    rating = max(1, min(5, rating))  # clamp 1-5

    # Try the real LLM first
    if load_llm():
        llm_review = generate_review(prompt)
        if llm_review:
            logger.info("Generated review using LoRA-finetuned Mistral 7B")
            return llm_review, float(rating)

    # Fallback: rule-based placeholder
    logger.info("Using placeholder review generator (LLM not available)")
    vocab = profile.domain_vocabulary
    has_pidgin = any(w in vocab for w in ["dey", "sabi", "na", "sef", "sha", "abeg", "wahala"])

    if rating >= 4:
        if has_pidgin:
            review = (
                f"This one na proper correct product. E dey work well well, "
                f"no wahala at all. For the price, e worth am."
            )
        else:
            review = (
                f"Very satisfied with this purchase. The quality is excellent "
                f"and it works exactly as described. Highly recommended."
            )
    elif rating == 3:
        if has_pidgin:
            review = (
                f"E just dey there sha. E no bad, e no too good. "
                f"Manageable for the price."
            )
        else:
            review = (
                f"It's an average product. Nothing spectacular but it gets "
                f"the job done. You get what you pay for."
            )
    else:
        if has_pidgin:
            review = (
                f"Abeg, this one no good at all o. Na waste of money. "
                f"The quality too poor, I no go recommend am."
            )
        else:
            review = (
                f"Very disappointed with this product. Poor quality and "
                f"doesn't match the description at all. Would not recommend."
            )

    return review, float(rating)


@router.post("/simulate-review", response_model=SimulateReviewResponse)
async def simulate_review(req: SimulateReviewRequest) -> SimulateReviewResponse:
    """
    Task A — Full pipeline for user modeling and review simulation.
    """
    profile_gen, temporal, naija_bert, prompt_template = _get_components()

    # ── Step 1-3: Extract user profile ────────────────────────────────
    # APG4RecSim internally calls Levenshtein normalizer + Pidgin VADER
    history_dicts = [
        {
            "item_id": r.item_id,
            "item_title": r.item_title,
            "item_category": r.item_category,
            "rating": r.rating,
            "review_text": r.review_text,
            "timestamp": r.timestamp,
        }
        for r in req.user_history
    ]

    profile = profile_gen.extract_profile(req.user_id, history_dicts)
    logger.info(f"Extracted profile for {req.user_id}: "
                f"price_sens={profile.price_sensitivity:.2f}, "
                f"quality={profile.quality_threshold:.2f}, "
                f"rating_skew={profile.rating_skew:.1f}")

    # ── Step 4: Embed normalized review texts with NaijaBERT ──────────
    review_texts = [r.review_text for r in req.user_history]
    embeddings = naija_bert.embed(review_texts)

    # ── Step 5: SASRec temporal attention ─────────────────────────────
    user_vector = temporal.encode_sequence(history_dicts, embeddings)
    logger.info(f"SASRec encoded user vector: norm={float(sum(user_vector**2)**0.5):.4f}")

    # ── Step 6: Build LLM prompt ──────────────────────────────────────
    profile_traits_str = profile_gen.to_prompt_string(profile)
    prompt = prompt_template.format(
        profile_traits=profile_traits_str,
        item_title=req.target_item.item_title,
        item_category=req.target_item.item_category,
        item_metadata=str(req.target_item.item_metadata),
        rating_skew=profile.rating_skew,
        verbosity_mean=profile.verbosity_mean,
    )

    # ── Step 7: Generate review ───────────────────────────────────────
    # PLACEHOLDER — swap _generate_review_placeholder with real LLM call
    generated_review, predicted_rating = _generate_review(prompt, profile)

    # ── Step 8: Build and return response ─────────────────────────────
    confidence = min(1.0, len(req.user_history) / 10.0)  # more history = more confident

    return SimulateReviewResponse(
        user_id=req.user_id,
        item_id=req.target_item.item_id,
        predicted_rating=predicted_rating,
        generated_review=generated_review,
        confidence=round(confidence, 2),
        profile_traits=ProfileTraits(
            price_sensitivity=round(profile.price_sensitivity, 3),
            quality_threshold=round(profile.quality_threshold, 3),
            rating_skew=round(profile.rating_skew, 2),
            dominant_vocabulary=profile.domain_vocabulary[:10],
        ),
    )
