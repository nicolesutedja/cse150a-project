"""
Unit tests for src/policies.py (Task 5).

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from env import play
from policies import (expected_information_gain, make_greedy_policy,
                      make_info_gain_policy, most_probable_policy)

# Same toy set as test_inference.py so answers can be checked by hand.
WORDS = ["crane", "crate", "craze", "grape", "slate"]
UNIFORM = {w: 1 / len(WORDS) for w in WORDS}


def test_greedy_picks_highest_letter_score_on_toy():
    """
    Letters per candidate: c3 r4 a5 e5 t2 n1 z1 g1 p1 s1 l1.
    crate = 3+4+5+2+5 = 19 is the unique best (crane/craze 18, grape 16, slate 14).
    """
    policy = make_greedy_policy(WORDS)
    assert policy(WORDS, UNIFORM, 1) == "crate"


def test_greedy_single_candidate_returns_it():
    policy = make_greedy_policy(WORDS + ["react"])
    assert policy(["slate"], {"slate": 1.0}, 3) == "slate"


def test_greedy_counts_repeated_guess_letters_once():
    """'eerie' must not get credit for its three e's."""
    policy = make_greedy_policy(["eerie", "crate"])
    # crate (c,r,a,t,e) beats eerie (e,r,i) on these candidates.
    assert policy(["crane", "crate", "craze"], {}, 1) == "crate"


def test_greedy_tie_prefers_candidate_then_alphabetical():
    """'react' ties 'crate' on score; the candidate wins the tie."""
    policy = make_greedy_policy(["react", "crate"] + WORDS)
    assert policy(WORDS, UNIFORM, 1) == "crate"


def test_greedy_returns_word_from_allowed_list():
    allowed = ["react", "crate", "slate"]
    policy = make_greedy_policy(allowed)
    assert policy(WORDS, UNIFORM, 1) in allowed


def test_greedy_does_not_mutate_inputs_and_is_deterministic():
    cands = list(WORDS)
    post = dict(UNIFORM)
    policy = make_greedy_policy(WORDS)
    first = policy(cands, post, 1)
    assert policy(cands, post, 1) == first
    assert cands == WORDS and post == UNIFORM


def test_greedy_plays_a_full_game_with_env_play():
    policy = make_greedy_policy(WORDS)
    for secret in WORDS:
        result = play(policy, secret, WORDS)
        assert result["won"] and result["guesses"] <= 6


# ---------------------------------------------------------------------------
# Most probable word
# ---------------------------------------------------------------------------
def test_most_probable_picks_highest_posterior():
    post = {"crane": 0.1, "crate": 0.2, "craze": 0.5, "grape": 0.1, "slate": 0.1}
    assert most_probable_policy(WORDS, post, 1) == "craze"


def test_most_probable_uniform_tie_goes_to_alphabetical_first():
    assert most_probable_policy(WORDS, UNIFORM, 1) == "crane"
    assert most_probable_policy(list(reversed(WORDS)), UNIFORM, 1) == "crane"


def test_most_probable_returns_a_candidate_and_does_not_mutate():
    cands = ["slate", "grape"]
    post = {"slate": 0.3, "grape": 0.7}
    assert most_probable_policy(cands, post, 2) == "grape"
    assert cands == ["slate", "grape"] and post == {"slate": 0.3, "grape": 0.7}


def test_most_probable_single_candidate():
    assert most_probable_policy(["slate"], {"slate": 1.0}, 4) == "slate"


def test_most_probable_plays_full_game_with_play_bayes_and_prior():
    from inference import play_bayes
    prior = {"crane": 0.1, "crate": 0.1, "craze": 0.6, "grape": 0.1, "slate": 0.1}
    # Prior favors craze, so a craze game should be won on guess 1.
    r = play_bayes(most_probable_policy, "craze", WORDS, prior=prior)
    assert r["won"] and r["guesses"] == 1
    for secret in WORDS:
        r = play_bayes(most_probable_policy, secret, WORDS, prior=prior)
        assert r["won"] and r["guesses"] <= 6


# ---------------------------------------------------------------------------
# Maximum expected information gain
# ---------------------------------------------------------------------------
def test_expected_information_gain_hand_checked_values():
    """crane splits the toy words into groups of size 1, 2, 1, 1:
    H = -(3 * 0.2*log2(0.2) + 0.4*log2(0.4)) = 1.9219 bits."""
    assert expected_information_gain("crane", WORDS, UNIFORM) == pytest.approx(1.9219281)
    skew = {"crane": 0.5, "crate": 0.1, "craze": 0.1, "grape": 0.2, "slate": 0.1}
    # groups {crane}=.5, {crate,craze}=.2, {grape}=.2, {slate}=.1
    assert expected_information_gain("crane", WORDS, skew) == pytest.approx(1.7609640)


def test_expected_information_gain_is_zero_when_guess_cannot_split():
    # crane gives the same feedback for crate and craze, so it learns nothing here.
    cands = ["crate", "craze"]
    post = {"crate": 0.5, "craze": 0.5}
    assert expected_information_gain("crane", cands, post) == pytest.approx(0.0)


def test_info_gain_toy_tie_goes_to_alphabetical_candidate():
    """crane, crate and craze all split the toy words equally (1.9219 bits)."""
    policy = make_info_gain_policy(WORDS)
    assert policy(WORDS, UNIFORM, 1) == "crane"


def test_info_gain_can_pick_a_non_candidate_that_splits_better():
    """'actin' gives 5 different feedbacks on the 5 toy words (log2(5) bits)."""
    policy = make_info_gain_policy(WORDS + ["actin"])
    assert policy(WORDS, UNIFORM, 1) == "actin"
    assert expected_information_gain("actin", WORDS, UNIFORM) == pytest.approx(2.3219281)


def test_info_gain_uses_the_posterior():
    skew = {"crane": 0.1, "crate": 0.5, "craze": 0.1, "grape": 0.15, "slate": 0.15}
    policy = make_info_gain_policy(WORDS)
    assert policy(WORDS, UNIFORM, 1) == "crane"
    assert policy(WORDS, skew, 1) == "crate"


def test_info_gain_single_candidate_returns_it():
    policy = make_info_gain_policy(WORDS + ["actin"])
    assert policy(["slate"], {"slate": 1.0}, 2) == "slate"


def test_info_gain_cache_does_not_mix_up_different_posteriors():
    skew = {"crane": 0.1, "crate": 0.5, "craze": 0.1, "grape": 0.15, "slate": 0.15}
    shared = make_info_gain_policy(WORDS)
    first_uniform = shared(WORDS, UNIFORM, 1)      # fills the cache
    assert shared(WORDS, skew, 1) == make_info_gain_policy(WORDS)(WORDS, skew, 1)
    assert shared(WORDS, UNIFORM, 1) == first_uniform


def test_info_gain_deterministic_and_does_not_mutate_inputs():
    cands, post = list(WORDS), dict(UNIFORM)
    policy = make_info_gain_policy(WORDS)
    assert policy(cands, post, 1) == policy(cands, post, 1)
    assert cands == WORDS and post == UNIFORM


def test_info_gain_plays_full_games_with_play_bayes():
    from inference import play_bayes
    policy = make_info_gain_policy(WORDS)
    for secret in WORDS:
        r = play_bayes(policy, secret, WORDS)
        assert r["won"] and r["guesses"] <= 6