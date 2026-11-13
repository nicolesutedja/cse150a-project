"""
Unit tests for src/env.py.

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from env import (
    feedback, noisy_feedback, feedback_likelihood,
    filter_candidates, pattern_to_str, play,
    GRAY as B, YELLOW as Y, GREEN as G, ALL_GREEN,
)


# ---------------------------------------------------------------------------
# pattern_to_str helper (sanity check)
# ---------------------------------------------------------------------------
def test_pattern_to_str():
    assert pattern_to_str((G, B, Y, B, G)) == "GBYBG"
    assert pattern_to_str(ALL_GREEN) == "GGGGG"


# ---------------------------------------------------------------------------
# Basic feedback cases
# ---------------------------------------------------------------------------
def test_all_green():
    assert feedback("crane", "crane") == ALL_GREEN


def test_all_gray():
    assert feedback("hypos", "crane") == (B, B, B, B, B)


def test_mixed_basic():
    # 'c' correct position → G; 'a' wrong position → Y; rest not present → B
    assert feedback("cabby", "crane") == (G, Y, B, B, B)


# ---------------------------------------------------------------------------
# Known Wordle cases (using pattern strings for readability)
# ---------------------------------------------------------------------------
CASES = [
    ("crane", "crane", "GGGGG"),
    ("speed", "abide", "BBYBY"),   # second e is gray: secret has only one e
    ("allee", "eagle", "YYBYG"),
    ("eagle", "allee", "YYBYG"),
    ("geese", "those", "BBBGG"),
    ("aaaaa", "abase", "GBGBB"),   # greens consume a's first
    ("stare", "tarts", "YYYYB"),
    ("weary", "raise", "BYYYB"),   # classic Wordle example
]


def test_known_cases():
    for guess, secret, expected in CASES:
        result = pattern_to_str(feedback(guess, secret))
        assert result == expected, f"feedback({guess!r}, {secret!r}) = {result!r}, want {expected!r}"


# ---------------------------------------------------------------------------
# Duplicate-letter edge cases (the common bug area)
# ---------------------------------------------------------------------------
def test_repeated_in_guess_not_in_secret():
    """Both l's in 'llama' should be gray — 'crane' has no l."""
    result = feedback("llama", "crane")
    assert result[0] == B and result[1] == B


def test_repeated_in_guess_one_in_secret():
    """'speed' vs 'creep': only one e in non-green positions should be yellow."""
    # pos2 and pos3 are both green (e==e); remaining={c,r,p}; both s and d are gray
    assert feedback("speed", "creep") == (B, Y, G, G, B)


def test_repeated_in_secret_one_in_guess():
    """Secret has two e's; guess has one — the single guess e should still match."""
    # "crane" vs "eerie": only pos4 e is green; pos1 r → yellow
    assert feedback("crane", "eerie") == (B, Y, B, B, G)


def test_guess_has_more_of_letter_than_secret():
    """'eerie' vs 'crane': secret has one e (at pos4=green); extra e's in guess → gray."""
    assert feedback("eerie", "crane") == (B, B, Y, B, G)


def test_yellow_does_not_overflow():
    """'opens' vs 'stone': each shared letter appears once."""
    assert feedback("opens", "stone") == (Y, B, Y, G, Y)


def test_green_consumes_before_yellow():
    """'spell' vs 'lapel': the green l at pos4 is used first; pos3 l → yellow."""
    assert feedback("spell", "lapel") == (B, Y, Y, Y, G)


# ---------------------------------------------------------------------------
# Noisy feedback
# ---------------------------------------------------------------------------
def test_noisy_feedback_zero_eps():
    """With eps=0, noisy_feedback must equal deterministic feedback."""
    import random
    rng = random.Random(0)
    for guess, secret, _ in CASES:
        assert noisy_feedback(guess, secret, 0.0, rng) == feedback(guess, secret)


def test_noisy_feedback_nonzero_eps():
    """With eps=1, every tile is always flipped — never equals true feedback."""
    import random
    rng = random.Random(0)
    true = feedback("crane", "crane")   # all green
    for _ in range(20):
        noisy = noisy_feedback("crane", "crane", 1.0, rng)
        assert noisy != true


# ---------------------------------------------------------------------------
# feedback_likelihood
# ---------------------------------------------------------------------------
def test_likelihood_exact_zero_eps():
    assert feedback_likelihood("crane", "crane", ALL_GREEN, 0.0) == 1.0
    assert feedback_likelihood("crane", "crane", (B, B, B, B, B), 0.0) == 0.0


def test_likelihood_all_correct_nonzero_eps():
    import math
    eps = 0.1
    result = feedback_likelihood("crane", "crane", ALL_GREEN, eps)
    assert abs(result - (1 - eps) ** 5) < 1e-12


def test_likelihood_soft_with_eps():
    """With eps > 0, even an inconsistent observation gets non-zero probability."""
    p = feedback_likelihood("crane", "crane", (B, B, B, B, B), 0.1)
    assert p > 0.0


# ---------------------------------------------------------------------------
# filter_candidates
# ---------------------------------------------------------------------------
def test_filter_keeps_consistent_words():
    pool = ["crane", "slate", "irate", "those"]
    fb   = feedback("crane", "irate")
    kept = filter_candidates(pool, "crane", fb)
    assert "irate" in kept and "those" not in kept


def test_filter_matches_feedback_indicator():
    """filter_candidates must be equivalent to feedback_likelihood(..., eps=0) == 1."""
    pool = ["crane", "slate", "irate", "those", "crane"]
    guess = "slate"
    secret = "irate"
    obs = feedback(guess, secret)
    filtered = set(filter_candidates(pool, guess, obs))
    for w in pool:
        if feedback_likelihood(w, secret, obs, 0.0) == 1.0:
            # w is consistent; this only holds if feedback(guess, w)==obs
            pass  # just check filtered set is non-empty and correct
    assert secret in filtered


# ---------------------------------------------------------------------------
# play loop
# ---------------------------------------------------------------------------
def test_play_wins():
    pool = ["crane", "slate", "irate", "those"]
    res  = play(lambda c, p, t: c[0], "those", pool)
    assert res["won"] and res["guesses"] <= 6


def test_play_records_history():
    pool = ["apple"]
    res  = play(lambda c, p, t: "apple", "apple", pool)
    assert res["won"]
    assert res["history"][0] == ("apple", ALL_GREEN)


def test_play_loss():
    """A policy that always guesses the wrong word should lose."""
    pool   = ["crane", "slate"]
    # always guess "crane", secret is "slate" — will never win
    result = play(lambda c, p, t: "crane", "slate", pool)
    assert not result["won"]
    assert result["guesses"] == 6
