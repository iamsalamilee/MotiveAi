"""Tests for Nigerian Pidgin VADER lexicon overrides."""
import pytest
from src.nlp.pidgin_vader import PidginVader

@pytest.fixture
def vader():
    return PidginVader()

def test_ginger_positive(vader):
    result = vader.score_sentiment("This thing dey ginger me well well")
    assert result["compound"] > 0.2, "ginger should map to positive sentiment"

def test_wash_negative(vader):
    result = vader.score_sentiment("Dem wash am bad bad")
    assert result["compound"] < 0.0, "wash should map to negative sentiment"

def test_hammer_positive(vader):
    result = vader.score_sentiment("E don hammer, the guy sharp")
    assert result["compound"] > 0.3, "hammer should map to positive"

def test_neutral_text(vader):
    result = vader.score_sentiment("I went to the restaurant today")
    assert isinstance(result["compound"], float)

def test_batch_score(vader):
    texts = ["E dey ginger me", "Dem wash am", "ordinary food"]
    results = vader.batch_score(texts)
    assert len(results) == 3
    assert all("compound" in r for r in results)
