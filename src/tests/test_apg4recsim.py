"""Tests for APG4RecSim profile generator."""
import pytest
from src.agent.apg4recsim import ProfileGenerator

SAMPLE_HISTORY = [
    {"item_id":"A1","item_title":"Headphones","item_category":"Electronics",
     "rating":4,"review_text":"E dey work well well, worth the money","timestamp":"2024-01-01T00:00:00Z"},
    {"item_id":"A2","item_title":"Laptop Stand","item_category":"Electronics",
     "rating":3,"review_text":"Okay product, nothing special but cheap and affordable","timestamp":"2024-03-01T00:00:00Z"},
    {"item_id":"A3","item_title":"USB Hub","item_category":"Electronics",
     "rating":5,"review_text":"This thing dey ginger me, e too good. Sharp product!","timestamp":"2024-05-01T00:00:00Z"},
]

@pytest.fixture
def generator():
    return ProfileGenerator()

def test_extract_profile_returns_profile(generator):
    profile = generator.extract_profile("U001", SAMPLE_HISTORY)
    assert profile.user_id == "U001"
    assert 1.0 <= profile.rating_skew <= 5.0
    assert 0.0 <= profile.price_sensitivity <= 1.0
    assert 0.0 <= profile.quality_threshold <= 1.0

def test_profile_vocab_non_empty(generator):
    profile = generator.extract_profile("U001", SAMPLE_HISTORY)
    assert len(profile.domain_vocabulary) > 0

def test_prompt_string_serializable(generator):
    profile = generator.extract_profile("U001", SAMPLE_HISTORY)
    prompt = generator.to_prompt_string(profile)
    assert isinstance(prompt, str)
    assert "price_sensitivity" in prompt

def test_empty_history_returns_default(generator):
    profile = generator.extract_profile("U999", [])
    assert profile.user_id == "U999"
    assert profile.rating_skew == 4.0
