"""
inference.py - Task 3: exact posterior (Juee)

Computes how likely each candidate word is to be the secret, given the
prior and every (guess, feedback) pair observed so far:

    posterior(w) = prior(w) * product over past guesses of P(feedback | w, guess)
                   ---------------------------------------------------------------
                                    sum of that over all candidates

With eps = 0 (noiseless), this is the same as filtering the candidate list.
"""

from env import (MAX_GUESSES, WordleEnv, feedback_likelihood,
                 filter_candidates)
from prior import normalize_posterior


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