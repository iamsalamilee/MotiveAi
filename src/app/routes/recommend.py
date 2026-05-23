from fastapi import APIRouter
from src.app.models.schemas import RecommendRequest, RecommendResponse, RecommendationItem

router = APIRouter()

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    """
    Task B — Recommendation.

    Pipeline (implement in order across Day 2-3):
      1. detect_cold_start(target_history)
      2. If cold-start: lasso.simulate(source_history, target_domain)
      3. Build NaijaBERT query vector from persona
      4. retriever.retrieve(query_vec, top_k=100)
      5. llamarec.rerank(profile, candidates)
      6. Return top req.top_k with reasoning strings
    """
    cold_start = len(req.user_persona.target_history) == 0

    # TODO: wire real pipeline
    return RecommendResponse(
        user_id=req.user_id,
        cold_start_detected=cold_start,
        cross_domain_used=cold_start,
        pseudo_interactions_generated=5 if cold_start else 0,
        recommendations=[
            RecommendationItem(
                rank=1, item_id="STUB_001",
                item_title="[STUB] Implement pipeline — see routes/recommend.py TODO",
                score=0.0, reasoning="Not yet implemented."
            )
        ]
    )
