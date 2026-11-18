"""Run exact Wordle MDP planning and Q-learning experiments.

From the repository root: ``python src/main.py --episodes 100000``
"""

import argparse

from ai import QLearner, WordleMDP


def run_mode(words, mode, episodes, report_every, eval_episodes, seed):
    print(f"\n=== Reward: {mode} ===")
    mdp = WordleMDP(words, reward_mode=mode)
    print(f"Reachable nonterminal states: {len(mdp.states)}")

    vi_values, vi_policy = mdp.value_iteration()
    pi_values, pi_policy = mdp.policy_iteration()
    vi_guesses = mdp.expected_guesses(vi_policy)
    pi_guesses = mdp.expected_guesses(pi_policy)
    max_value_difference = max(abs(vi_values[s] - pi_values[s]) for s in mdp.states)
    policy_agreement = sum(vi_policy[s] == pi_policy[s] for s in vi_policy)
    print(f"Value iteration start value: {vi_values[mdp.start]:.8f}")
    print(f"Policy iteration start value: {pi_values[mdp.start]:.8f}")
    print(f"Start value difference: {abs(vi_values[mdp.start] - pi_values[mdp.start]):.3g}")
    print(f"Maximum state-value difference: {max_value_difference:.3g}")
    print(f"Policy action agreement: {policy_agreement}/{len(vi_policy)} states")
    print(f"Expected guesses (VI / PI): {vi_guesses:.4f} / {pi_guesses:.4f}")
    print(f"Expected guess difference: {abs(vi_guesses - pi_guesses):.3g}")

    learner = QLearner(mdp, seed=seed)
    reports = learner.train(episodes, report_every, eval_episodes)
    close_episode = None
    for episode, average in reports:
        gap = average - vi_guesses
        print(f"Q episode {episode:>8}: avg guesses={average:.4f}, gap from VI={gap:+.4f}")
        if close_episode is None and abs(gap) <= 0.05:
            close_episode = episode
    if close_episode is None:
        print("Q-learning did not come within 0.05 guesses at the recorded checkpoints.")
    else:
        print(f"First recorded checkpoint within 0.05 guesses: {close_episode} episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--report-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--modes", nargs="+", choices=("standard", "sparse", "entropy"),
                        default=("standard", "sparse", "entropy"))
    args = parser.parse_args()
    words = WordleMDP().words
    print(f"Vocabulary size: {len(words)}")
    for offset, mode in enumerate(args.modes):
        run_mode(words, mode, args.episodes, args.report_every,
                 args.eval_episodes, args.seed + offset)


if __name__ == "__main__":
    main()
