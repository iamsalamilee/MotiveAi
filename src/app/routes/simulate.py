from fastapi import APIRouter
from src.app.models.schemas import SimulateReviewRequest, SimulateReviewResponse, ProfileTraits

router = APIRouter()

@router.post("/simulate-review", response_model=SimulateReviewResponse)
async def simulate_review(req: SimulateReviewRequest) -> SimulateReviewResponse:
    """
    Task A — User Modeling.

    Pipeline (implement in order across Day 2-3):
      1. Pidgin VADER normalization on history texts
      2. Levenshtein spelling normalization
      3. NaijaBERT embedding of normalized history
      4. APG4RecSim profile extraction
      5. SASRec temporal attention over sequence
      6. Build generation prompt from profile + target item
      7. LoRA LLM conditioned generation
      8. Parse rating from output
    """
    # TODO: wire real pipeline
    return SimulateReviewResponse(
        user_id=req.user_id,
        item_id=req.target_item.item_id,
        predicted_rating=4.0,
        generated_review="[STUB] Implement pipeline — see routes/simulate.py TODO",
        confidence=0.0,
        profile_traits=ProfileTraits(
            price_sensitivity=0.5,
            quality_threshold=0.7,
            rating_skew=4.0,
            dominant_vocabulary=[]
        )
    )
