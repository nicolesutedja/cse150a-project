"""
inference.py - Task 3: exact posterior (Juee)

Computes how likely each candidate word is to be the secret, given the
prior and every (guess, feedback) pair observed so far:

    posterior(w) = prior(w) * product over past guesses of P(feedback | w, guess)
                   ---------------------------------------------------------------
                                    sum of that over all candidates

With eps = 0 (noiseless), this is the same as filtering the candidate list.
"""

from collections import Counter
from itertools import accumulate
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt

from env import (
    MAX_GUESSES,
    WordleEnv,
    feedback_likelihood,
    filter_candidates,
    load_guesses,
    load_vocab100,
    noisy_feedback,
)
from prior import build_priors, normalize_posterior

def likelihood_weighted_priors(candidates, prior, history, eps=0.0):
    """
    Score each candidate by prior x likelihood of all past feedback.

    The scores are NOT normalized (they don't sum to 1). Use
    exact_posterior() to get probabilities.

    Parameters
    ----------
    candidates : list[str]
        Words that could be the secret.
    prior : dict[str, float]
        P(w) for each candidate, e.g. uniform {w: 1/len(words)} or
        {w: 2 ** model.log_prob(w)} from prior.py.
    history : list[tuple[str, tuple]]
        (guess, observed_feedback) pairs in the order they were played,
        e.g. [("crate", (2, 2, 2, 0, 2))].
    eps : float
        Chance that each tile is misreported. 0 = feedback is always correct,
        so words that don't match get a score of exactly 0.

    Returns
    -------
    dict[str, float]
        {word: unnormalized score}
    """
    scores = {}
    for w in candidates:
        score = prior[w]
        for guess, observed in history:
            score *= feedback_likelihood(guess, w, observed, eps)
        scores[w] = score
    return scores


def exact_posterior(candidates, prior, history, eps=0.0):
    """
    Exact posterior over the candidates, given all feedback so far.

    Parameters are the same as likelihood_weighted_priors().

    Returns
    -------
    dict[str, float]
        {word: probability}, summing to 1. If every score is 0 (only
        possible with eps = 0 and contradictory feedback), normalize_posterior
        falls back to a uniform distribution.
    """
    scores = likelihood_weighted_priors(candidates, prior, history, eps)
    return normalize_posterior(None, candidates, prior_weights=scores)


def play_bayes(choose_guess, secret, pool, prior=None, eps=0.0,
               guess_list=None, max_guesses=MAX_GUESSES):
    """
    Same as env.play(), but the policy gets the exact posterior instead of a
    uniform one, and the game can have noisy feedback.

    Parameters
    ----------
    choose_guess : function(candidates, posterior, guess_number) -> word
    secret : str
        The real secret word (only the game uses it).
    pool : list[str]
        Words that could be the secret.
    prior : dict[str, float] or None
        P(w) for each word in pool. None = uniform.
    eps : float
        Noise level for the game's feedback and for the posterior.

    Returns
    -------
    dict with won (bool), guesses (int), history (list of (guess, feedback)).
    """
    if prior is None:
        prior = {w: 1.0 / len(pool) for w in pool}
    env = WordleEnv(pool, guess_list, max_guesses, eps=eps)
    env.reset(secret)
    candidates = list(pool)
    history = []

    for t in range(1, max_guesses + 1):
        posterior = exact_posterior(candidates, prior, history, eps)
        guess = choose_guess(candidates, posterior, t)
        fb, done, won = env.step(guess)
        history.append((guess, fb))
        if won:
            return {"won": True, "guesses": t, "history": history}
        if eps == 0:
            # Safe to cross words out only when feedback can't be wrong.
            candidates = filter_candidates(candidates, guess, fb)
        if done:
            break

    return {"won": False, "guesses": max_guesses, "history": history}

"""
approx_inference.py — Task 4: Approximate inference (Dishita)

Implements:
  1. Rejection sampling with noisy feedback simulation
  2. Likelihood weighting using the noise model likelihood
  3. Total variation distance (TVD) against exact_posterior
  4. Sweeps over eps and sample sizes, saving a runtime table and error plot
"""

def _sample_from_prior(candidates: list[str], prior: dict[str, float],
                       rng: random.Random) -> str:
    """Sample one candidate word proportional to prior[w]."""
    weights = [max(0.0, prior.get(w, 0.0)) for w in candidates]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(candidates)
    r = rng.uniform(0.0, total)
    cumulative = 0.0
    for w, wt in zip(candidates, weights):
        cumulative += wt
        if r <= cumulative:
            return w
    return candidates[-1]


def rejection_sampling(candidates: list[str], prior: dict[str, float],
                       history: list[tuple[str, tuple[int, ...]]],
                       eps: float, num_samples: int,
                       seed: int | None = None) -> tuple[dict[str, float], dict]:
    """
    Estimate posterior via rejection sampling.

    Samples w from the prior, simulates noisy feedback for every past guess
    in history, and keeps w only if all simulated feedbacks match observed.
    """
    rng = random.Random(seed)
    counts = Counter()
    accepted = 0
    t0 = time.perf_counter()

    for _ in range(num_samples):
        w = _sample_from_prior(candidates, prior, rng)
        matched_all = True
        for guess, observed_fb in history:
            sim_fb = noisy_feedback(guess, w, eps, rng=rng)
            if sim_fb != observed_fb:
                matched_all = False
                break
        if matched_all:
            counts[w] += 1
            accepted += 1

    runtime = time.perf_counter() - t0
    acceptance_rate = accepted / num_samples if num_samples > 0 else 0.0

    if accepted > 0:
        posterior = {w: counts[w] / accepted for w in candidates}
    else:
        posterior = {w: 1.0 / len(candidates) for w in candidates}

    metrics = {
        "runtime_sec": runtime,
        "acceptance_rate": acceptance_rate,
        "accepted": accepted,
        "num_samples": num_samples,
    }
    return posterior, metrics


def likelihood_weighting(candidates: list[str], prior: dict[str, float],
                         history: list[tuple[str, tuple[int, ...]]],
                         eps: float, num_samples: int,
                         seed: int | None = None) -> tuple[dict[str, float], dict]:
    """
    Estimate posterior via likelihood weighting.

    Samples w from the prior and weights each draw by the product of
    P(F_t | w, G_t, eps) across all past guesses.
    """
    rng = random.Random(seed)
    weighted_counts = Counter()
    t0 = time.perf_counter()

    for _ in range(num_samples):
        w = _sample_from_prior(candidates, prior, rng)
        weight = 1.0
        for guess, observed_fb in history:
            weight *= feedback_likelihood(guess, w, observed_fb, eps)
        if weight > 0.0:
            weighted_counts[w] += weight

    runtime = time.perf_counter() - t0
    total_weight = sum(weighted_counts.values())

    if total_weight > 0.0:
        posterior = {w: weighted_counts[w] / total_weight for w in candidates}
    else:
        posterior = {w: 1.0 / len(candidates) for w in candidates}

    metrics = {
        "runtime_sec": runtime,
        "num_samples": num_samples,
    }
    return posterior, metrics


def total_variation_distance(p_exact: dict[str, float],
                             p_approx: dict[str, float],
                             candidates: list[str]) -> float:
    """Compute TVD = 0.5 * sum_w |P_exact(w) - P_approx(w)|."""
    return 0.5 * sum(
        abs(p_exact.get(w, 0.0) - p_approx.get(w, 0.0)) for w in candidates
    )


def run_sweep(candidates: list[str], prior: dict[str, float],
              history: list[tuple[str, tuple[int, ...]]],
              eps_values=(0.01, 0.05, 0.1),
              sample_sizes=(100, 1000, 10000),
              seed: int = 42,
              results_dir: Path | None = None):
    """
    Sweep eps and sample sizes, compare against exact_posterior, print a
    runtime & error table, and save an error-vs-sample-size plot.
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for eps_idx, eps in enumerate(eps_values):
        p_exact = exact_posterior(candidates, prior, history, eps)
        for n_idx, n in enumerate(sample_sizes):
            run_seed = seed + eps_idx * 100 + n_idx
            rej_post, rej_m = rejection_sampling(
                candidates, prior, history, eps, n, seed=run_seed
            )
            lw_post, lw_m = likelihood_weighting(
                candidates, prior, history, eps, n, seed=run_seed
            )
            rows.append({
                "eps": eps,
                "samples": n,
                "rej_tvd": total_variation_distance(p_exact, rej_post, candidates),
                "lw_tvd": total_variation_distance(p_exact, lw_post, candidates),
                "rej_acc": rej_m["acceptance_rate"],
                "rej_time": rej_m["runtime_sec"],
                "lw_time": lw_m["runtime_sec"],
            })

    header = (
        f"{'eps':>5} {'N':>7} {'Rej TVD':>9} {'LW TVD':>9} "
        f"{'Rej Acc%':>9} {'Rej Time(s)':>12} {'LW Time(s)':>11}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['eps']:>5.2f} {r['samples']:>7d} {r['rej_tvd']:>9.4f} "
            f"{r['lw_tvd']:>9.4f} {r['rej_acc'] * 100:>8.2f}% "
            f"{r['rej_time']:>12.4f} {r['lw_time']:>11.4f}"
        )

    # One plot of error (TVD) vs. sample size per method across eps values
    plt.figure(figsize=(8, 5))
    for eps in eps_values:
        sub = [r for r in rows if r["eps"] == eps]
        ns = [r["samples"] for r in sub]
        plt.plot(
            ns,
            [r["rej_tvd"] for r in sub],
            marker="o",
            label=f"Rejection sampling (eps={eps})",
        )
        plt.plot(
            ns,
            [r["lw_tvd"] for r in sub],
            marker="s",
            linestyle="--",
            label=f"Likelihood weighting (eps={eps})",
        )

    plt.xscale("log")
    plt.xlabel("Sample size (N)")
    plt.ylabel("Total Variation Distance (TVD)")
    plt.title("Approximate Inference Error vs. Sample Size")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = results_dir / "task4_tvd_vs_samples.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {plot_path}")
    return rows


if __name__ == "__main__":
    vocab = load_vocab100()
    priors = build_priors(load_guesses())
    prior_dist = normalize_posterior(priors["per_position"], vocab)
    sample_history = [("crane", (0, 1, 0, 0, 2))]
    run_sweep(vocab, prior_dist, sample_history)