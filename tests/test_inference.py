"""
Unit tests for src/inference.py (Task 3).

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from env import feedback, filter_candidates, load_vocab100, play
from inference import (
    exact_posterior,
    likelihood_weighted_priors,
    likelihood_weighting,
    play_bayes,
    rejection_sampling,
    total_variation_distance,
)
# Small toy setup so results are easy to check by hand.
WORDS = ["crane", "crate", "craze", "grape", "slate"]
UNIFORM = {w: 1 / len(WORDS) for w in WORDS}
SECRET = "crane"


def history_for(guesses, secret=SECRET):
    """Clean (noise-free) feedback history for a list of guesses."""
    return [(g, feedback(g, secret)) for g in guesses]


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.0, 0.05, 0.2])
def test_posterior_sums_to_one(eps):
    post = exact_posterior(WORDS, UNIFORM, history_for(["crate"]), eps)
    assert sum(post.values()) == pytest.approx(1.0)


def test_no_history_returns_prior():
    """With no guesses yet, the posterior is just the normalized prior."""
    prior = {"crane": 4, "crate": 3, "craze": 1, "grape": 1, "slate": 1}
    post = exact_posterior(WORDS, prior, history=[], eps=0.0)
    assert post["crane"] == pytest.approx(0.4)
    assert post["slate"] == pytest.approx(0.1)


def test_prior_changes_ranking():
    """crane and craze fit the feedback equally; the prior breaks the tie."""
    prior = {"crane": 0.1, "crate": 0.1, "craze": 0.6, "grape": 0.1, "slate": 0.1}
    post = exact_posterior(WORDS, prior, history_for(["crate"]), eps=0.0)
    assert post["craze"] > post["crane"] > 0


# ---------------------------------------------------------------------------
# Noiseless case (eps = 0) must equal filtering
# ---------------------------------------------------------------------------
def test_eps0_toy_matches_filter():
    hist = history_for(["crate"])
    post = exact_posterior(WORDS, UNIFORM, hist, eps=0.0)
    kept = {w for w, p in post.items() if p > 0}
    assert kept == set(filter_candidates(WORDS, *hist[0]))
    assert kept == {"crane", "craze"}
    assert post["crane"] == pytest.approx(0.5)


def test_eps0_vocab_matches_filter():
    words = load_vocab100()
    prior = {w: 1 / len(words) for w in words}
    secret = words[17]
    hist = history_for([words[3], words[50]], secret)
    expected = list(words)
    for g, fb in hist:
        expected = filter_candidates(expected, g, fb)
    post = exact_posterior(words, prior, hist, eps=0.0)
    assert {w for w, p in post.items() if p > 0} == set(expected)


def test_unnormalized_scores_are_0_or_prior_when_noiseless():
    scores = likelihood_weighted_priors(WORDS, UNIFORM, history_for(["crate"]), 0.0)
    assert scores["grape"] == 0
    assert scores["crane"] == pytest.approx(UNIFORM["crane"])


# ---------------------------------------------------------------------------
# Noisy case (eps > 0)
# ---------------------------------------------------------------------------
def test_noisy_true_word_ranks_first():
    hist = history_for(["crate", "slate"])
    post = exact_posterior(WORDS, UNIFORM, hist, eps=0.05)
    assert max(post, key=post.get) in {"crane", "craze"}
    assert post["crane"] > post["grape"]


def test_noisy_mismatch_is_small_but_not_zero():
    """With noise, a word that doesn't fit is unlikely, not impossible."""
    post = exact_posterior(WORDS, UNIFORM, history_for(["crate"]), eps=0.05)
    assert 0 < post["grape"] < post["crane"]


# ---------------------------------------------------------------------------
# play_bayes
# ---------------------------------------------------------------------------
def test_play_bayes_eps0_matches_play():
    words = load_vocab100()
    top = lambda candidates, posterior, t: max(posterior, key=posterior.get)
    for secret in words[:20]:
        assert play_bayes(top, secret, words)["guesses"] == \
               play(top, secret, words)["guesses"]


# ---------------------------------------------------------------------------
# Approximate inference: rejection sampling & likelihood weighting (Task 4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.01, 0.05, 0.1])
def test_approx_posteriors_valid_distributions_and_metrics(eps):
    hist = history_for(["crate"])
    p_rej, rej_metrics = rejection_sampling(
        WORDS, UNIFORM, hist, eps=eps, num_samples=500, seed=42
    )
    p_lw, lw_metrics = likelihood_weighting(
        WORDS, UNIFORM, hist, eps=eps, num_samples=500, seed=42
    )

    assert set(p_rej.keys()) == set(WORDS)
    assert set(p_lw.keys()) == set(WORDS)
    assert sum(p_rej.values()) == pytest.approx(1.0)
    assert sum(p_lw.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in p_rej.values())
    assert all(0.0 <= v <= 1.0 for v in p_lw.values())
    assert 0.0 <= rej_metrics["acceptance_rate"] <= 1.0
    assert rej_metrics["runtime_sec"] >= 0.0
    assert lw_metrics["runtime_sec"] >= 0.0


def test_tvd_identities():
    hist = history_for(["crate"])
    p_exact = exact_posterior(WORDS, UNIFORM, hist, eps=0.05)
    assert total_variation_distance(p_exact, p_exact, WORDS) == pytest.approx(0.0)

    tvd = total_variation_distance(p_exact, UNIFORM, WORDS)
    assert 0.0 <= tvd <= 1.0


def test_approx_no_history_matches_prior():
    prior = {"crane": 0.4, "crate": 0.3, "craze": 0.1, "grape": 0.1, "slate": 0.1}
    p_exact = exact_posterior(WORDS, prior, history=[], eps=0.05)
    p_rej, rej_metrics = rejection_sampling(
        WORDS, prior, history=[], eps=0.05, num_samples=4000, seed=7
    )
    p_lw, lw_metrics = likelihood_weighting(
        WORDS, prior, history=[], eps=0.05, num_samples=4000, seed=7
    )

    assert rej_metrics["acceptance_rate"] == pytest.approx(1.0)
    assert lw_metrics["runtime_sec"] >= 0.0
    assert total_variation_distance(p_exact, p_rej, WORDS) < 0.05
    assert total_variation_distance(p_exact, p_lw, WORDS) < 0.05


def test_likelihood_weighting_error_decreases_with_samples():
    hist = history_for(["crate"])
    eps = 0.05
    p_exact = exact_posterior(WORDS, UNIFORM, hist, eps=eps)

    p_lw_100, _ = likelihood_weighting(
        WORDS, UNIFORM, hist, eps=eps, num_samples=100, seed=11
    )
    p_lw_5000, _ = likelihood_weighting(
        WORDS, UNIFORM, hist, eps=eps, num_samples=5000, seed=11
    )

    assert total_variation_distance(p_exact, p_lw_5000, WORDS) < (
        total_variation_distance(p_exact, p_lw_100, WORDS)
    )


def test_rejection_sampling_zero_samples_fallback():
    hist = history_for(["crate"])
    p_rej, rej_metrics = rejection_sampling(
        WORDS, UNIFORM, hist, eps=0.0, num_samples=0, seed=99
    )
    assert rej_metrics["acceptance_rate"] == 0.0
    assert sum(p_rej.values()) == pytest.approx(1.0)