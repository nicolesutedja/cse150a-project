"""
prior.py — Task 2: Priors by MLE (Nicole)

Three models fit on the guess list with add-one (Laplace) smoothing:
  1. letter_freq  : P(w) = P(l1)*P(l2)*P(l3)*P(l4)*P(l5), one shared dist
  2. per_position : P(w) = P1(l1)*...*P5(l5), separate dist per position
  3. pos_bigram   : P(w) = P(l1)*P(l2|l1)*P(l3|l2)*P(l4|l3)*P(l5|l4)

Evaluation: average bits per word = mean(-log2 P(w)) over a word list.
Uniform baseline: 5 * log2(26) ≈ 23.5 bits.

Public interface
----------------
  build_priors(guess_words)         -> dict of {name: model_object}
  score_word(model, word)           -> log2 probability (negative = bits)
  normalize_posterior(model, candidates, prior_weights=None) -> dict[str, float]
  compare_lists(guess_words, solution_words) -> dict of analysis results
  evaluate_model(model, word_list)  -> float (avg bits per word)
"""

import math
import string
from collections import Counter
from pathlib import Path

ALPHABET = list(string.ascii_lowercase)   # 26 letters
WORD_LEN  = 5


# ---------------------------------------------------------------------------
# Internal model classes
# ---------------------------------------------------------------------------

class LetterFreqModel:
    """P(w) = prod_i P(letter_i), one shared unigram distribution."""

    def __init__(self, words):
        counts = Counter()
        for w in words:
            counts.update(w)
        total = sum(counts.values()) + 26          # add-one smoothing
        self.log_p = {c: math.log2((counts[c] + 1) / total) for c in ALPHABET}

    def log_prob(self, word):
        return sum(self.log_p[c] for c in word)


class PerPositionModel:
    """P(w) = prod_i P_i(letter_i), separate distribution per position."""

    def __init__(self, words):
        counts = [Counter() for _ in range(WORD_LEN)]
        for w in words:
            for i, c in enumerate(w):
                counts[i][c] += 1
        self.log_p = []
        for pos_counts in counts:
            total = sum(pos_counts.values()) + 26
            self.log_p.append(
                {c: math.log2((pos_counts[c] + 1) / total) for c in ALPHABET}
            )

    def log_prob(self, word):
        return sum(self.log_p[i][c] for i, c in enumerate(word))


class PosBigramModel:
    """P(w) = P(l1) * prod_{i=1..4} P(l_{i+1} | l_i)."""

    def __init__(self, words):
        # unigram for position 0
        uni_counts = Counter(w[0] for w in words)
        uni_total  = sum(uni_counts.values()) + 26
        self.log_p0 = {c: math.log2((uni_counts[c] + 1) / uni_total)
                       for c in ALPHABET}

        # bigram counts: bigram[c_prev][c_next]
        bigram = {c: Counter() for c in ALPHABET}
        for w in words:
            for i in range(WORD_LEN - 1):
                bigram[w[i]][w[i + 1]] += 1

        # conditional log-probs with add-one smoothing
        self.log_cond = {}
        for prev in ALPHABET:
            total = sum(bigram[prev].values()) + 26
            self.log_cond[prev] = {
                c: math.log2((bigram[prev][c] + 1) / total) for c in ALPHABET
            }

    def log_prob(self, word):
        lp = self.log_p0[word[0]]
        for i in range(1, WORD_LEN):
            lp += self.log_cond[word[i - 1]][word[i]]
        return lp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_priors(guess_words: list) -> dict:
    """
    Fit all three models on guess_words.

    Returns
    -------
    dict with keys 'letter_freq', 'per_position', 'pos_bigram'
    each mapping to a model object with a .log_prob(word) method.
    """
    return {
        "letter_freq":   LetterFreqModel(guess_words),
        "per_position":  PerPositionModel(guess_words),
        "pos_bigram":    PosBigramModel(guess_words),
    }


def score_word(model, word: str) -> float:
    """Return log2 probability of word under model (always <= 0)."""
    return model.log_prob(word)


def evaluate_model(model, word_list: list) -> float:
    """Average bits per word = mean(-log2 P(w)) over word_list."""
    return -sum(model.log_prob(w) for w in word_list) / len(word_list)


def normalize_posterior(model, candidates: list,
                        prior_weights: dict = None) -> dict:
    """
    Compute a normalized posterior over candidates.

    If prior_weights is None, uses the model's probabilities (in linear space).
    Returns dict[word -> probability] summing to 1.
    """
    if prior_weights is not None:
        raw = {w: prior_weights[w] for w in candidates}
    else:
        raw = {w: 2 ** model.log_prob(w) for w in candidates}
    total = sum(raw.values())
    if total == 0:
        # fallback to uniform
        return {w: 1.0 / len(candidates) for w in candidates}
    return {w: v / total for w, v in raw.items()}


# ---------------------------------------------------------------------------
# List comparison (per-position distributions + KL divergence)
# ---------------------------------------------------------------------------

def _position_distributions(words: list) -> list:
    """Return 5 normalized letter distributions (one per position)."""
    dists = []
    for pos in range(WORD_LEN):
        counts = Counter(w[pos] for w in words)
        total  = sum(counts.values())
        dists.append({c: counts.get(c, 0) / total for c in ALPHABET})
    return dists


def _kl_divergence(p: dict, q: dict, eps: float = 1e-9) -> float:
    """KL(P || Q) in bits."""
    kl = 0.0
    for c in ALPHABET:
        pc = p.get(c, 0) + eps
        qc = q.get(c, 0) + eps
        kl += pc * math.log2(pc / qc)
    return kl


def compare_lists(guess_words: list, solution_words: list) -> dict:
    """
    Compare per-position letter distributions between the two lists.

    Returns a dict with:
      'guess_pos_dists'    : list of 5 dicts (letter -> freq)
      'solution_pos_dists' : list of 5 dicts
      'kl_per_position'    : list of 5 KL(solutions || guesses) values in bits
      'avg_kl'             : mean KL across positions
      'guess_ending_s'     : fraction of guess words ending in 's'
      'solution_ending_s'  : fraction of solution words ending in 's'
    """
    g_dists = _position_distributions(guess_words)
    s_dists = _position_distributions(solution_words)
    kl_vals = [_kl_divergence(s_dists[i], g_dists[i]) for i in range(WORD_LEN)]

    return {
        "guess_pos_dists":    g_dists,
        "solution_pos_dists": s_dists,
        "kl_per_position":    kl_vals,
        "avg_kl":             sum(kl_vals) / WORD_LEN,
        "guess_ending_s":     sum(1 for w in guess_words    if w[-1] == "s") / len(guess_words),
        "solution_ending_s":  sum(1 for w in solution_words if w[-1] == "s") / len(solution_words),
    }


# ---------------------------------------------------------------------------
# Results table (called from __main__ or notebook)
# ---------------------------------------------------------------------------

def print_results_table(guess_words: list, solution_words: list):
    """Print the full comparison + evaluation table to stdout."""
    UNIFORM_BITS = 5 * math.log2(26)

    print("=" * 60)
    print("TASK 2: MLE PRIOR ANALYSIS")
    print("=" * 60)

    # --- list comparison ---
    comp = compare_lists(guess_words, solution_words)
    print(f"\nWord-list sizes")
    print(f"  Guesses:   {len(guess_words):>6}")
    print(f"  Solutions: {len(solution_words):>6}")

    print(f"\nEnding-letter distribution (last position = 's')")
    print(f"  Guess list:    {comp['guess_ending_s']:.3f}  ({comp['guess_ending_s']*100:.1f}%)")
    print(f"  Solution list: {comp['solution_ending_s']:.3f}  ({comp['solution_ending_s']*100:.1f}%)")
    print(f"  → Solutions end in 's' far less often — the solutions list")
    print(f"    was curated to avoid plural forms.")

    print(f"\nKL divergence KL(solutions || guesses) per position (bits)")
    for i, kl in enumerate(comp['kl_per_position']):
        print(f"  Position {i+1}: {kl:.4f} bits")
    print(f"  Average:   {comp['avg_kl']:.4f} bits")
    print(f"  → Non-zero KL means the distributions differ; fitting a prior")
    print(f"    on the guess list is an approximation for solutions.")

    # --- model evaluation ---
    models = build_priors(guess_words)

    print(f"\nModel evaluation: average bits per word  (lower = better)")
    print(f"  Uniform baseline: {UNIFORM_BITS:.2f} bits")
    print(f"  {'Model':<20} {'on guesses':>12} {'on solutions':>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*14}")
    for name, model in models.items():
        g_bits = evaluate_model(model, guess_words)
        s_bits = evaluate_model(model, solution_words)
        print(f"  {name:<20} {g_bits:>12.3f} {s_bits:>14.3f}")

    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from env import load_guesses, load_solutions

    guess_words    = load_guesses()
    solution_words = load_solutions()
    print_results_table(guess_words, solution_words)
