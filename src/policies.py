"""
policies.py -- Task 5: non-RL guessing policies 

Every policy follows the team interface:

    choose_guess(candidates, posterior, guess_number) -> word
      candidates   : list[str]        words still consistent with the feedback
      posterior    : dict[str, float] probability of each candidate (sums to 1)
      guess_number : int              1-indexed turn number

Policies that need the allowed-guess list (any word the player may type, which
can include words that are no longer candidates) are built by a factory that
captures it, e.g. make_greedy_policy(guess_list).
"""

import math
from collections import Counter, defaultdict

from env import feedback


def make_greedy_policy(guess_list):
    """
    Letter-frequency greedy.

    1. Count, for each letter, how many of the current candidates contain it
       (a letter is counted once per candidate word).
    2. Score each allowed guess by the sum of those counts over the guess's
       DISTINCT letters (a repeated letter in the guess only counts once).
    3. Return the highest-scoring guess.

    Ties are broken deterministically: prefer a word that is still a candidate
    (it could be the answer), then alphabetical order.

    The posterior is ignored on purpose -- this policy only looks at letters.
    """
    allowed = sorted(set(guess_list))          # sorted => alphabetical tie-break

    def choose_guess(candidates, posterior, guess_number):
        if len(candidates) == 1:
            return candidates[0]

        letter_count = Counter()
        for word in candidates:
            letter_count.update(set(word))

        candidate_set = set(candidates)

        def score(guess):
            return (sum(letter_count[c] for c in set(guess)),   # main score
                    guess in candidate_set)                     # tie-break 1

        # max() keeps the FIRST best item, and `allowed` is sorted, so any
        # remaining tie goes to the alphabetically earliest word.
        return max(allowed, key=score)

    return choose_guess


def most_probable_policy(candidates, posterior, guess_number):
    """
    Most probable word: guess the candidate with the highest posterior.

    Needs no factory because it only ever picks from `candidates`.
    Ties (e.g. a uniform posterior) go to the alphabetically earliest word,
    so results are reproducible.
    """
    return max(sorted(candidates), key=lambda w: posterior.get(w, 0.0))


# ---------------------------------------------------------------------------
# Maximum expected information gain
# ---------------------------------------------------------------------------
def expected_information_gain(guess, candidates, posterior):
    """
    H(guess) = -sum_k p_k * log2(p_k) in bits, where the candidates are grouped
    by the feedback pattern `guess` would produce and p_k is the posterior mass
    of group k (masses are renormalized to sum to 1 over `candidates`).
    """
    mass = defaultdict(float)
    for w in candidates:
        mass[feedback(guess, w)] += posterior.get(w, 0.0)
    total = sum(mass.values())
    if total <= 0.0:                      # no posterior information: uniform
        mass = defaultdict(float)
        for w in candidates:
            mass[feedback(guess, w)] += 1.0
        total = float(len(candidates))
    return -sum((m / total) * math.log2(m / total) for m in mass.values() if m > 0)


def make_info_gain_policy(guess_list):
    """
    Maximum expected information gain.

    For every allowed guess, compute expected_information_gain() and pick the
    largest. The guess may be a non-candidate if it splits the candidates best.

    Ties are broken like the greedy policy: prefer a word that is still a
    candidate, then alphabetical. Entropies are rounded to 9 decimals first so
    float noise cannot decide a tie.

    Speed: the cost per decision is len(guess_list) * len(candidates) feedback
    calls. Policies are deterministic, so each decision is cached by
    (candidates, posterior); in particular, turn 1 is computed once and reused
    for every game.
    """
    allowed = sorted(set(guess_list))
    cache = {}

    def choose_guess(candidates, posterior, guess_number):
        if len(candidates) == 1:
            return candidates[0]

        key = (tuple(candidates), tuple(posterior.get(w, 0.0) for w in candidates))
        if key in cache:
            return cache[key]

        candidate_set = set(candidates)

        def score(guess):
            return (round(expected_information_gain(guess, candidates, posterior), 9),
                    guess in candidate_set)

        best = max(allowed, key=score)          # first max => alphabetical
        cache[key] = best
        return best

    return choose_guess