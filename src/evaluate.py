"""
evaluate.py -- Task 5 / Task 7: run policies on many targets and summarize.

evaluate() works with ANY game loop, so the same code serves the main
comparison (uniform posterior), the prior ablation, and the noisy runs:

    play_fn = lambda policy, secret: play(policy, secret, vocab, vocab)
    play_fn = lambda policy, secret: play_bayes(policy, secret, vocab,
                                                prior=prior, guess_list=vocab)
    row = evaluate(policy, vocab, play_fn)

Counting rule: a game that is not won within max_guesses counts as a failure
and as max_guesses guesses in the average (play() already reports it that way).
"""

import csv
from collections import Counter
from pathlib import Path

from env import MAX_GUESSES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate(policy, targets, play_fn, max_guesses=MAX_GUESSES):
    """
    Play `policy` once against every secret in `targets`.

    play_fn(policy, secret) -> dict with 'won', 'guesses', 'history'.

    Returns a dict:
      n, wins, win_rate, avg_guesses (losses count as max_guesses),
      avg_guesses_wins_only, worst_case, distribution ({1..max_guesses, 'fail'}),
      repeated_guess_games (games where the same word was guessed twice)
    """
    dist = Counter()
    total = 0
    win_total = 0
    wins = 0
    worst = 0
    repeats = 0

    for secret in targets:
        r = play_fn(policy, secret)
        total += r["guesses"]
        worst = max(worst, r["guesses"])
        if r["won"]:
            wins += 1
            win_total += r["guesses"]
            dist[r["guesses"]] += 1
        else:
            dist["fail"] += 1
        guessed = [g for g, _ in r["history"]]
        if len(guessed) != len(set(guessed)):
            repeats += 1

    n = len(targets)
    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n,
        "avg_guesses": total / n,
        "avg_guesses_wins_only": (win_total / wins) if wins else float("nan"),
        "worst_case": worst,
        "distribution": {**{k: dist[k] for k in range(1, max_guesses + 1)},
                         "fail": dist["fail"]},
        "repeated_guess_games": repeats,
    }


def format_table(rows, max_guesses=MAX_GUESSES):
    """rows: list of (name, evaluate() result). Returns a printable string."""
    head = (f"{'policy':<22}{'win%':>7}{'avg':>7}{'worst':>7}"
            + "".join(f"{k:>5}" for k in range(1, max_guesses + 1))
            + f"{'fail':>6}{'repeats':>9}")
    lines = [head, "-" * len(head)]
    for name, r in rows:
        d = r["distribution"]
        lines.append(
            f"{name:<22}{r['win_rate'] * 100:>6.1f}%{r['avg_guesses']:>7.2f}"
            f"{r['worst_case']:>7d}"
            + "".join(f"{d[k]:>5d}" for k in range(1, max_guesses + 1))
            + f"{d['fail']:>6d}{r['repeated_guess_games']:>9d}"
        )
    return "\n".join(lines)


def save_csv(rows, path, max_guesses=MAX_GUESSES, setting=None):
    """Write one CSV row per policy so the team can build report tables.
    If `setting` is given, it is written in an extra first column so the CSV
    says which experiment it came from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = (([] if setting is None else ["setting"]) +
            ["policy", "n", "wins", "win_rate", "avg_guesses",
             "avg_guesses_wins_only", "worst_case"]
            + [f"guesses_{k}" for k in range(1, max_guesses + 1)]
            + ["fail", "repeated_guess_games"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, r in rows:
            d = r["distribution"]
            w.writerow(([] if setting is None else [setting]) +
                       [name, r["n"], r["wins"], round(r["win_rate"], 4),
                        round(r["avg_guesses"], 4),
                        round(r["avg_guesses_wins_only"], 4), r["worst_case"]]
                       + [d[k] for k in range(1, max_guesses + 1)]
                       + [d["fail"], r["repeated_guess_games"]])


if __name__ == "__main__":
    import argparse
    import time

    from env import load_guesses, load_solutions, load_vocab100, play
    from policies import (make_greedy_policy, make_info_gain_policy,
                          most_probable_policy)

    parser = argparse.ArgumentParser(description="Evaluate the Task 5 policies.")
    parser.add_argument("--full", action="store_true",
                        help="run on the full word lists instead of the 100-word vocabulary")
    parser.add_argument("--all-targets", action="store_true",
                        help="with --full: use every solution as a secret (slow) "
                             "instead of the 100 vocabulary words")
    args = parser.parse_args()

    if args.full:
        # Candidates start as all solutions; any valid word may be guessed.
        pool = load_solutions()
        allowed = sorted(set(load_guesses()) | set(pool))
        targets = pool if args.all_targets else load_vocab100()
        pool_set = set(pool)
        assert all(w in pool_set for w in targets), "every target must be a solution"
        which = "all solutions" if args.all_targets else "the 100 vocab words"
        setting = f"full_list(targets={which}), uniform posterior, no noise"
        title = ("FULL LISTS (NOT COMPARABLE TO RL): "
                 f"{len(pool)} candidates, {len(allowed)} allowed guesses, "
                 f"secrets = {which}")
        out = RESULTS_DIR / ("task5_full_list_all_targets_policies.csv"
                             if args.all_targets else "task5_full_list_policies.csv")
    else:
        pool = allowed = targets = load_vocab100()
        setting = "vocab100, uniform posterior, no noise"
        title = "100-word vocabulary, uniform posterior, no noise"
        out = RESULTS_DIR / "task5_vocab100_policies.csv"

    policies = {
        "letter_freq_greedy": make_greedy_policy(allowed),
        "most_probable": most_probable_policy,
        "info_gain": make_info_gain_policy(allowed),
    }

    # Same evaluate() and same play() for every setting.
    play_fn = lambda policy, secret: play(policy, secret, pool, allowed)
    rows = []
    for name, policy in policies.items():
        print(f"running {name} on {len(targets)} secrets ...", flush=True)
        t0 = time.perf_counter()
        rows.append((name, evaluate(policy, targets, play_fn)))
        print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n" + title)
    print(format_table(rows))
    save_csv(rows, out, setting=setting)
    print(f"\nSaved {out}")