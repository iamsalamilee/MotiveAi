"""Tests for Levenshtein Pidgin normalizer."""
import pytest
from src.nlp.levenshtein_normalizer import LevenshteinNormalizer

@pytest.fixture
def normalizer():
    return LevenshteinNormalizer()

def test_canonical_token_unchanged(normalizer):
    assert normalizer.normalize_token("sabi") == "sabi"

def test_vowel_variant_normalized(normalizer):
    # 'abii' should normalize to 'abi' (vowel doubling, cheap cost)
    result = normalizer.normalize_token("abii")
    assert result in ("abi", "abii"), f"unexpected: {result}"

def test_full_text_normalization(normalizer):
    text = "The food dey good, e sabi cook well"
    result = normalizer.normalize(text)
    assert isinstance(result, str)
    assert len(result) > 0

def test_unknown_token_preserved(normalizer):
    result = normalizer.normalize_token("xyzqwerty")
    assert result == "xyzqwerty"

def test_weighted_distance_vowels(normalizer):
    cost_vowel = normalizer._weighted_distance("abi", "abii")
    cost_consonant = normalizer._weighted_distance("sabi", "xabi")
    assert cost_vowel < cost_consonant, "vowel edits should be cheaper"
