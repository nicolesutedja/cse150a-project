"""
Unit tests for src/prior.py (Task 2).

Run with:  python -m pytest tests/test_prior.py -v
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prior import (
    build_priors, score_word, evaluate_model,
    normalize_posterior, compare_lists,
    LetterFreqModel, PerPositionModel, PosBigramModel,
    ALPHABET,
)

TINY_CORPUS = ["crane", "slate", "irate", "those", "abide", "eagle"]


# ---------------------------------------------------------------------------
# Model construction: probabilities must be valid
# ---------------------------------------------------------------------------

def test_log_probs_are_nonpositive():
    """All log2 probabilities must be <= 0 (probability <= 1)."""
    models = build_priors(TINY_CORPUS)
    for word in TINY_CORPUS:
        for name, model in models.items():
            assert model.log_prob(word) <= 0, f"{name}: log_prob({word}) > 0"


def test_probs_sum_to_at_most_one():
    """Sum of P(w) over a small closed set must not exceed 1."""
    models = build_priors(TINY_CORPUS)
    for name, model in models.items():
        total = sum(2 ** model.log_prob(w) for w in TINY_CORPUS)
        assert total <= 1.0 + 1e-9, f"{name}: probabilities sum to {total:.4f} > 1"


# ---------------------------------------------------------------------------
# Smoothing: every letter must have non-zero probability
# ---------------------------------------------------------------------------

def test_smoothing_covers_all_letters():
    """Add-one smoothing: even unseen letters must have p > 0."""
    # corpus has no 'z' — but log_prob of a word with z must still be finite
    models = build_priors(["aaaaa"])
    for name, model in models.items():
        lp = model.log_prob("zzzzz")
        assert math.isfinite(lp), f"{name}: log_prob('zzzzz') is not finite"
        assert lp < 0, f"{name}: log_prob('zzzzz') >= 0"


# ---------------------------------------------------------------------------
# Relative ordering: common patterns should outscore rare ones
# ---------------------------------------------------------------------------

def test_per_position_common_beats_rare():
    """'crane' seen in corpus should outscore an unseen all-z word."""
    model = PerPositionModel(TINY_CORPUS)
    assert model.log_prob("crane") > model.log_prob("zzzzz")


def test_bigram_common_beats_rare():
    model = PosBigramModel(TINY_CORPUS)
    assert model.log_prob("crane") > model.log_prob("zzzzz")


# ---------------------------------------------------------------------------
# evaluate_model: must beat uniform baseline on the training corpus
# ---------------------------------------------------------------------------

def test_evaluate_beats_uniform():
    """Every model should use fewer bits per word than the uniform baseline."""
    uniform_bits = 5 * math.log2(26)   # ≈ 23.50
    models = build_priors(TINY_CORPUS)
    for name, model in models.items():
        bits = evaluate_model(model, TINY_CORPUS)
        assert bits < uniform_bits, (
            f"{name}: {bits:.3f} bits >= uniform {uniform_bits:.3f}")


# ---------------------------------------------------------------------------
# normalize_posterior
# ---------------------------------------------------------------------------

def test_posterior_sums_to_one():
    models = build_priors(TINY_CORPUS)
    for name, model in models.items():
        post = normalize_posterior(model, TINY_CORPUS)
        total = sum(post.values())
        assert abs(total - 1.0) < 1e-9, f"{name}: posterior sums to {total}"


def test_posterior_all_positive():
    models = build_priors(TINY_CORPUS)
    for name, model in models.items():
        post = normalize_posterior(model, TINY_CORPUS)
        for w, p in post.items():
            assert p > 0, f"{name}: posterior[{w}] = {p}"


def test_posterior_subsets_correctly():
    """Posterior over a subset should ignore words outside the subset."""
    model = PerPositionModel(TINY_CORPUS)
    subset = ["crane", "slate"]
    post = normalize_posterior(model, subset)
    assert set(post.keys()) == set(subset)
    assert abs(sum(post.values()) - 1.0) < 1e-9


def test_posterior_with_external_weights():
    """When prior_weights are supplied, they override model probabilities."""
    model = PerPositionModel(TINY_CORPUS)
    weights = {w: 1.0 for w in TINY_CORPUS}   # uniform weights
    post = normalize_posterior(model, TINY_CORPUS, prior_weights=weights)
    # all words should have equal posterior
    vals = list(post.values())
    assert all(abs(v - vals[0]) < 1e-9 for v in vals)


# ---------------------------------------------------------------------------
# compare_lists
# ---------------------------------------------------------------------------

def test_compare_lists_keys():
    comp = compare_lists(TINY_CORPUS, TINY_CORPUS)
    for key in ("guess_pos_dists", "solution_pos_dists",
                "kl_per_position", "avg_kl",
                "guess_ending_s", "solution_ending_s"):
        assert key in comp, f"Missing key: {key}"


def test_kl_self_is_zero():
    """KL(P || P) should be near zero when both lists are identical."""
    comp = compare_lists(TINY_CORPUS, TINY_CORPUS)
    for i, kl in enumerate(comp["kl_per_position"]):
        assert abs(kl) < 1e-6, f"Position {i}: KL(self||self) = {kl}"


def test_kl_nonnegative():
    """KL divergence is always >= 0."""
    comp = compare_lists(TINY_CORPUS, ["crane", "slate", "irate"])
    for kl in comp["kl_per_position"]:
        assert kl >= -1e-9, f"Negative KL: {kl}"


def test_ending_s_fraction():
    words_with_s    = ["crass", "slabs", "those"]
    words_without_s = ["crane", "slate", "irate"]
    comp = compare_lists(words_with_s, words_without_s)
    assert abs(comp["guess_ending_s"]    - 2/3) < 1e-9
    assert abs(comp["solution_ending_s"] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Bigram vs unigram: bigram should fit training data better
# ---------------------------------------------------------------------------

def test_bigram_lower_bits_than_letter_freq():
    """Position-bigram is strictly richer than letter_freq — expect lower bits."""
    models = build_priors(TINY_CORPUS)
    bigram_bits  = evaluate_model(models["pos_bigram"],  TINY_CORPUS)
    lettfreq_bits = evaluate_model(models["letter_freq"], TINY_CORPUS)
    assert bigram_bits < lettfreq_bits, (
        f"pos_bigram ({bigram_bits:.3f}) not better than letter_freq ({lettfreq_bits:.3f})")
