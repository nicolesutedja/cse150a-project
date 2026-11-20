"""
Unit tests for src/evaluate.py.

Run with:  python -m pytest tests/ -v
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from env import play
from evaluate import evaluate, format_table, save_csv
from policies import most_probable_policy

WORDS = ["crane", "crate", "craze", "grape", "slate"]


def play_fn(policy, secret):
    return play(policy, secret, WORDS, WORDS)


def test_evaluate_known_distribution_most_probable():
    """With alphabetical tie-break the guess counts are fixed:
    crane 1, crate 2, craze 3, grape 2, slate 2."""
    r = evaluate(most_probable_policy, WORDS, play_fn)
    assert r["n"] == 5 and r["wins"] == 5
    assert r["win_rate"] == pytest.approx(1.0)
    assert r["avg_guesses"] == pytest.approx(10 / 5)
    assert r["worst_case"] == 3
    assert r["distribution"][1] == 1
    assert r["distribution"][2] == 3
    assert r["distribution"][3] == 1
    assert r["distribution"]["fail"] == 0
    assert r["repeated_guess_games"] == 0


def test_evaluate_counts_failures_as_six_and_flags_repeats():
    always_crane = lambda candidates, posterior, t: "crane"
    r = evaluate(always_crane, WORDS, play_fn)
    assert r["wins"] == 1 and r["distribution"]["fail"] == 4
    assert r["distribution"][1] == 1
    assert r["avg_guesses"] == pytest.approx((1 + 4 * 6) / 5)
    assert r["worst_case"] == 6
    assert r["repeated_guess_games"] == 4
    assert r["win_rate"] == pytest.approx(0.2)


def test_distribution_adds_up_to_n():
    r = evaluate(most_probable_policy, WORDS, play_fn)
    assert sum(r["distribution"].values()) == r["n"]


def test_format_table_and_save_csv(tmp_path):
    rows = [("most_probable", evaluate(most_probable_policy, WORDS, play_fn))]
    text = format_table(rows)
    assert "most_probable" in text and "win%" in text
    out = tmp_path / "out.csv"
    save_csv(rows, out)
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert got[0]["policy"] == "most_probable"
    assert float(got[0]["avg_guesses"]) == pytest.approx(2.0)
    assert int(got[0]["guesses_2"]) == 3


def test_save_csv_setting_column_is_optional(tmp_path):
    rows = [("most_probable", evaluate(most_probable_policy, WORDS, play_fn))]
    with_setting = tmp_path / "a.csv"
    without = tmp_path / "b.csv"
    save_csv(rows, with_setting, setting="toy")
    save_csv(rows, without)
    with open(with_setting, newline="") as f:
        got = list(csv.DictReader(f))[0]
    assert got["setting"] == "toy" and got["policy"] == "most_probable"
    with open(without, newline="") as f:
        assert "setting" not in next(csv.reader(f))