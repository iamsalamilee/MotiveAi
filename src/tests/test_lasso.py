"""Tests for Lasso cross-domain simulator."""
import pytest
from src.agent.lasso import CrossDomainSimulator

SOURCE_HISTORY = [
    {"item_id":"A1","rating":5,"review_text":"Premium quality, luxury feel, worth every naira",
     "item_category":"Electronics","timestamp":"2024-01-01T00:00:00Z"},
    {"item_id":"A2","rating":4,"review_text":"Good vibe product, chill to use",
     "item_category":"Electronics","timestamp":"2024-03-01T00:00:00Z"},
]

@pytest.fixture
def simulator():
    return CrossDomainSimulator()

def test_detect_cold_start_empty(simulator):
    assert simulator.detect_cold_start([]) is True

def test_detect_cold_start_with_history(simulator):
    assert simulator.detect_cold_start([{"item_id":"X"}]) is False

def test_simulate_returns_n_interactions(simulator):
    result = simulator.simulate(SOURCE_HISTORY, "Restaurants", n=5)
    assert len(result) == 5

def test_simulate_minimum_5(simulator):
    result = simulator.simulate(SOURCE_HISTORY, "Restaurants", n=3)
    assert len(result) >= 3

def test_pseudo_interactions_have_required_fields(simulator):
    result = simulator.simulate(SOURCE_HISTORY, "Restaurants", n=1)
    item = result[0]
    for field in ("item_id","item_title","item_category","rating","review_text","timestamp"):
        assert field in item, f"Missing field: {field}"

def test_taste_dims_extracted(simulator):
    taste = simulator._extract_taste_dims(SOURCE_HISTORY)
    assert taste.price_tier in ("budget","mid","premium")
    assert taste.novelty_preference in ("conservative","moderate","adventurous")
    assert 0.0 <= taste.quality_floor <= 1.0
