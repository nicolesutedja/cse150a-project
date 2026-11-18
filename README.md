# Wordle MDP and Reinforcement Learning

This project models Wordle as a finite Markov decision process over a fixed
vocabulary of 100 possible secret words. The rules and feedback calculation are
in `src/env.py`; the planning and learning algorithms are in `src/ai.py`.

## `src/ai.py`

`WordleMDP` builds the MDP. A state contains the sorted tuple of remaining
candidate secrets and the number of guesses already used. Available actions are
the 100 words in the vocabulary. For each guess, the transition model groups
candidate secrets by their deterministic Wordle feedback and assigns each
feedback group probability `group size / candidate count`.

The class enumerates reachable states from the initial state, then provides:

- Value iteration and policy iteration for the exact optimal value and policy.
- Expected guesses under a policy, counting a loss as using all six guesses.
- Three reward modes: `standard` (−1 per guess and an extra −10 for a final
  miss), `sparse` (+1 for a win and 0 otherwise), and `entropy` (−1 per guess
  plus the reduction in candidate-set entropy).

`QLearner` implements tabular, epsilon-greedy Q-learning with discount factor
1, a decaying exploration rate, and a visit-count-based learning rate. It
evaluates its learned policy at regular episode checkpoints.

## `src/main.py`

The command-line runner performs exact planning and Q-learning for each selected
reward mode. It reports the reachable-state count, value-iteration and policy-
iteration agreement, expected guesses, and the learned policy's average guesses
and gap from the value-iteration policy at each checkpoint.

Run all reward modes with the defaults:

```bash
python src/main.py
```

Choose run settings or reward modes with flags:

```bash
python src/main.py --episodes 100000 --report-every 5000 --eval-episodes 1000
python src/main.py --modes standard
python src/main.py --modes entropy sparse --seed 12
```

Options:

- `--episodes`: Q-learning training episodes per reward mode.
- `--report-every`: training interval between evaluation checkpoints.
- `--eval-episodes`: games sampled to evaluate the learned policy at each
  checkpoint.
- `--seed`: random seed for repeatability.
- `--modes`: one or more of `standard`, `sparse`, or `entropy`.

The vocabulary is loaded from `data/vocab100.txt`. The exact state enumeration
can take some time before the first results appear.


## Reading the results

The standard reward makes maximizing return equivalent to minimizing expected
guesses while penalizing a six-guess loss. The sparse reward only distinguishes
wins from losses, so its optimal policy is not required to minimize the number
of guesses. For that mode, compare win performance rather than treating its
expected-guess count as an optimal speed target. Evaluation averages are based
on sampled games and may vary slightly between checkpoints.
