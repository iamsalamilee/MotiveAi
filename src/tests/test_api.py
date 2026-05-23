"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

SAMPLE_REVIEW_REQUEST = {
    "user_id": "U001",
    "user_history": [{
        "item_id": "A1", "item_title": "Test Item", "item_category": "Electronics",
        "rating": 4, "review_text": "E dey work well", "timestamp": "2024-01-01T00:00:00Z"
    }],
    "target_item": {
        "item_id": "B1", "item_title": "New Item", "item_category": "Electronics",
        "item_metadata": {"brand": "Sony"}
    }
}

SAMPLE_RECOMMEND_REQUEST = {
    "user_id": "U001",
    "user_persona": {
        "source_domain": "Electronics",
        "source_history": [{
            "item_id": "A1", "item_title": "Test Item", "item_category": "Electronics",
            "rating": 4, "review_text": "E dey work well", "timestamp": "2024-01-01T00:00:00Z"
        }],
        "target_domain": "Restaurants",
        "target_history": [],
        "conversational_context": ["quiet place with parking"]
    },
    "top_k": 5
}

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_simulate_review_returns_200():
    resp = client.post("/simulate-review", json=SAMPLE_REVIEW_REQUEST)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "U001"
    assert "predicted_rating" in data
    assert "generated_review" in data

def test_simulate_review_empty_history_rejected():
    req = {**SAMPLE_REVIEW_REQUEST, "user_history": []}
    resp = client.post("/simulate-review", json=req)
    assert resp.status_code == 422

def test_recommend_returns_200():
    resp = client.post("/recommend", json=SAMPLE_RECOMMEND_REQUEST)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "U001"
    assert data["cold_start_detected"] is True
    assert len(data["recommendations"]) > 0

def test_recommend_top_k_respected():
    req = {**SAMPLE_RECOMMEND_REQUEST, "top_k": 3}
    resp = client.post("/recommend", json=req)
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) <= 3
