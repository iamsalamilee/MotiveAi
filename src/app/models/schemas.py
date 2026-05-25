from pydantic import BaseModel, Field

class ReviewItem(BaseModel):
    item_id: str = Field(..., examples=["B001"])
    item_title: str = Field(..., examples=["Samsung Galaxy S23"])
    item_category: str = Field(..., examples=["Electronics"])
    rating: int = Field(ge=1, le=5, examples=[4])
    review_text: str = Field(..., examples=["This phone dey work well, battery strong."])
    timestamp: str = Field(..., examples=["2024-03-15T14:30:00Z"])

class TargetItem(BaseModel):
    item_id: str = Field(..., examples=["B002"])
    item_title: str = Field(..., examples=["Sony Headphones"])
    item_category: str = Field(..., examples=["Electronics"])
    item_metadata: dict = Field(default={}, examples=[{"brand": "Sony", "color": "Black"}])

class SimulateReviewRequest(BaseModel):
    user_id: str = Field(..., examples=["U12345"])
    user_history: list[ReviewItem] = Field(min_length=1)
    target_item: TargetItem

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "U12345",
                    "user_history": [
                        {
                            "item_id": "B001",
                            "item_title": "Samsung Galaxy S23",
                            "item_category": "Electronics",
                            "rating": 4,
                            "review_text": "This phone dey work well, battery strong.",
                            "timestamp": "2024-03-15T14:30:00Z"
                        }
                    ],
                    "target_item": {
                        "item_id": "B002",
                        "item_title": "Sony Headphones",
                        "item_category": "Electronics",
                        "item_metadata": {"brand": "Sony", "color": "Black"}
                    }
                }
            ]
        }
    }

class ProfileTraits(BaseModel):
    price_sensitivity: float
    quality_threshold: float
    rating_skew: float
    dominant_vocabulary: list[str]

class SimulateReviewResponse(BaseModel):
    user_id: str
    item_id: str
    predicted_rating: float
    generated_review: str
    confidence: float
    profile_traits: ProfileTraits

class UserPersona(BaseModel):
    source_domain: str = Field(..., examples=["Electronics"])
    source_history: list[ReviewItem] = []
    target_domain: str = Field(..., examples=["Restaurants"])
    target_history: list[ReviewItem] = []
    conversational_context: list[str] = Field(default=[], examples=[["Looking for a quiet place", "Must have parking"]])

class RecommendRequest(BaseModel):
    user_id: str = Field(..., examples=["U12345"])
    user_persona: UserPersona
    top_k: int = Field(default=10, ge=1, le=50)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "U12345",
                    "user_persona": {
                        "source_domain": "Electronics",
                        "source_history": [
                            {
                                "item_id": "B001",
                                "item_title": "Samsung Galaxy S23",
                                "item_category": "Electronics",
                                "rating": 4,
                                "review_text": "This phone dey work well, battery strong.",
                                "timestamp": "2024-03-15T14:30:00Z"
                            }
                        ],
                        "target_domain": "Restaurants",
                        "target_history": [],
                        "conversational_context": [
                            "Looking for a quiet place",
                            "Must have parking"
                        ]
                    },
                    "top_k": 5
                }
            ]
        }
    }

class RecommendationItem(BaseModel):
    rank: int
    item_id: str
    item_title: str
    score: float
    reasoning: str

class RecommendResponse(BaseModel):
    user_id: str
    cold_start_detected: bool
    cross_domain_used: bool
    pseudo_interactions_generated: int
    recommendations: list[RecommendationItem]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    chroma_ready: bool
    items_indexed: int
