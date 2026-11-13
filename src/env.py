"""
env.py — Shared Wordle environment foundation (Task 1)

Tiles: 0 = gray, 1 = yellow, 2 = green.
Feedback is a tuple of 5 ints, e.g. (2, 0, 1, 0, 0).

Terms used in the project:
  guess list     = words the agent may guess (valid guesses)
  candidate pool = words that can be the secret (solutions, or the 100-word vocab)
  candidates     = pool words still consistent with all feedback so far
"""

import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAY, YELLOW, GREEN = 0, 1, 2
WORD_LEN    = 5
MAX_GUESSES = 6
ALL_GREEN   = (GREEN,) * WORD_LEN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = Path(__file__).parent.parent / "data"
GUESSES_PATH   = DATA_DIR / "valid_guesses.txt"
SOLUTIONS_PATH = DATA_DIR / "valid_solutions.txt"
VOCAB100_PATH  = DATA_DIR / "vocab100.txt"

VOCAB100_SEED  = 42


# ---------------------------------------------------------------------------
# Feedback functions
# ---------------------------------------------------------------------------
def feedback(guess: str, secret: str) -> tuple:
    """
    Two-pass Wordle feedback that handles duplicate letters correctly.

    Pass 1 — mark exact matches green; count remaining (non-green) secret letters.
    Pass 2 — for each non-green position, mark yellow if that letter still has
              remaining count > 0 (then decrement), otherwise gray.

    Returns a 5-tuple of ints: GRAY=0, YELLOW=1, GREEN=2.
    """
    result    = [GRAY] * WORD_LEN
    remaining = {}
    for i in range(WORD_LEN):
        if guess[i] == secret[i]:
            result[i] = GREEN
        else:
            remaining[secret[i]] = remaining.get(secret[i], 0) + 1
    for i in range(WORD_LEN):
        if result[i] == GREEN:
            continue
        if remaining.get(guess[i], 0) > 0:
            result[i] = YELLOW
            remaining[guess[i]] -= 1
    return tuple(result)


def noisy_feedback(guess: str, secret: str, eps: float, rng=random) -> tuple:
    """Each tile is independently misreported with probability eps
    (flipped to one of the other two colors uniformly)."""
    true_fb = feedback(guess, secret)
    out = []
    for tile in true_fb:
        if rng.random() < eps:
            out.append(rng.choice([t for t in (GRAY, YELLOW, GREEN) if t != tile]))
        else:
            out.append(tile)
    return tuple(out)


def feedback_likelihood(guess: str, secret: str, observed: tuple, eps: float) -> float:
    """
    P(observed feedback | secret, guess) under the eps noise model.
    With eps = 0 this is the 0/1 indicator (exact match only).
    Used by Juee's exact posterior and Dishita's approximate inference.
    """
    true_fb = feedback(guess, secret)
    p = 1.0
    for obs, true in zip(observed, true_fb):
        p *= (1.0 - eps) if obs == true else (eps / 2.0)
    return p


def filter_candidates(candidates: list, guess: str, observed: tuple) -> list:
    """Noiseless exact inference: keep only words consistent with the feedback."""
    return [w for w in candidates if feedback(guess, w) == observed]


def pattern_to_str(fb: tuple) -> str:
    """Convert an int-tuple feedback to a human-readable string, e.g. 'GBYYG'."""
    return "".join("BYG"[t] for t in fb)


# ---------------------------------------------------------------------------
# Word-list helpers
# ---------------------------------------------------------------------------
def load_words(path) -> list:
    """Read one word per line (or a CSV with a 'word' column).
    Returns lowercase 5-letter alphabetic words only."""
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().split(",")[0].strip().lower()
            if len(w) == WORD_LEN and w.isalpha():
                words.append(w)
    return words


def load_guesses() -> list:
    return load_words(GUESSES_PATH)


def load_solutions() -> list:
    return load_words(SOLUTIONS_PATH)


# ---------------------------------------------------------------------------
# 100-word reduced vocabulary (fixed seed — generate once, never regenerate)
# ---------------------------------------------------------------------------
def _sample_vocab100(solutions: list, seed: int = VOCAB100_SEED) -> list:
    return sorted(random.Random(seed).sample(solutions, 100))


def load_vocab100() -> list:
    """Load the saved 100-word vocabulary; raises if it doesn't exist yet."""
    if not VOCAB100_PATH.exists():
        raise FileNotFoundError(
            f"{VOCAB100_PATH} not found. Run generate_vocab100() once to create it."
        )
    return load_words(VOCAB100_PATH)


def generate_vocab100(solutions: list = None) -> list:
    """
    Create data/vocab100.txt exactly once with a fixed seed.
    Raises RuntimeError if the file already exists to prevent accidental
    regeneration mid-experiment (which would invalidate saved results).
    """
    if VOCAB100_PATH.exists():
        raise RuntimeError(
            f"{VOCAB100_PATH} already exists. Delete it manually if you really "
            "want to regenerate — this will invalidate existing results."
        )
    if solutions is None:
        solutions = load_solutions()
    vocab = _sample_vocab100(solutions)
    VOCAB100_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB100_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    print(f"Saved {len(vocab)} words to {VOCAB100_PATH}")
    return vocab


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class WordleEnv:
    """
    Stateful game environment.

    env.reset(secret=None)       -> secret word (random if not given)
    env.step(guess)              -> (feedback_tuple, done, won)
    """

    def __init__(self, pool, guess_list=None, max_guesses=MAX_GUESSES,
                 eps=0.0, seed=None):
        self.pool        = list(pool)
        self.guess_list  = list(guess_list) if guess_list is not None else list(pool)
        self.max_guesses = max_guesses
        self.eps         = eps
        self.rng         = random.Random(seed)
        self.secret      = None
        self.guess_number = 0

    def reset(self, secret=None):
        self.secret       = secret if secret is not None else self.rng.choice(self.pool)
        self.guess_number = 0
        return self.secret

    def step(self, guess: str):
        self.guess_number += 1
        fb   = (noisy_feedback(guess, self.secret, self.eps, self.rng)
                if self.eps > 0 else feedback(guess, self.secret))
        won  = (fb == ALL_GREEN)
        done = won or self.guess_number >= self.max_guesses
        return fb, done, won


# ---------------------------------------------------------------------------
# Play loop (used by Tasks 5, 6, 7)
# ---------------------------------------------------------------------------
def play(choose_guess, secret: str, pool: list, guess_list: list = None,
         max_guesses: int = MAX_GUESSES) -> dict:
    """
    Play one noiseless game with any policy.

    choose_guess(candidates, posterior, guess_number) -> word
      candidates   : list[str]  words still consistent with all feedback
      posterior    : dict[str, float]  uniform over candidates (replace with
                     your prior-weighted posterior by wrapping this loop)
      guess_number : int  1-indexed turn number

    Returns
    -------
    dict with keys:
      won      (bool)
      guesses  (int)   equals max_guesses on a loss
      history  list of (guess_str, feedback_tuple)
    """
    env        = WordleEnv(pool, guess_list, max_guesses)
    env.reset(secret)
    candidates = list(pool)
    history    = []

    for t in range(1, max_guesses + 1):
        posterior = {w: 1.0 / len(candidates) for w in candidates}
        guess     = choose_guess(candidates, posterior, t)
        fb, done, won = env.step(guess)
        history.append((guess, fb))
        if won:
            return {"won": True, "guesses": t, "history": history}
        candidates = filter_candidates(candidates, guess, fb)
        if done:
            break

    return {"won": False, "guesses": max_guesses, "history": history}
