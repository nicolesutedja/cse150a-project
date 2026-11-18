"""Exact planning and tabular Q-learning for the 100-word Wordle MDP."""

from collections import defaultdict, deque
import math
import random

from env import ALL_GREEN, MAX_GUESSES, feedback, load_vocab100


def entropy(candidates):
    """Entropy in bits for the uniform distribution over candidates."""
    return math.log2(len(candidates)) if candidates else 0.0


class WordleMDP:
    """Finite MDP with states ``(sorted candidate tuple, guesses used)``."""

    def __init__(self, words=None, reward_mode="standard"):
        self.words = tuple(sorted(words or load_vocab100()))
        self.reward_mode = reward_mode
        self.start = (self.words, 0)
        self.actions = self.words
        self._groups = {}
        self.states = self.enumerate_states()

    def groups(self, candidates, guess):
        key = (candidates, guess)
        if key not in self._groups:
            grouped = defaultdict(list)
            for secret in candidates:
                grouped[feedback(guess, secret)].append(secret)
            self._groups[key] = {
                pattern: tuple(words) for pattern, words in grouped.items()
            }
        return self._groups[key]

    def transitions(self, state, guess):
        candidates, turn = state
        groups = self.groups(candidates, guess)
        result = []
        for pattern, subset in groups.items():
            won = pattern == ALL_GREEN
            done = won or turn + 1 >= MAX_GUESSES
            next_state = None if done else (subset, turn + 1)
            probability = len(subset) / len(candidates)
            reward = self.reward(state, guess, subset, won, done)
            result.append((probability, next_state, reward, len(subset), won))
        return result

    def reward(self, state, guess, subset, won, done):
        if self.reward_mode == "standard":
            return -1.0 - (10.0 if done and not won else 0.0)
        if self.reward_mode == "sparse":
            return 1.0 if won else 0.0
        if self.reward_mode == "entropy":
            before = entropy(state[0])
            after = 0.0 if won else entropy(subset)
            return -1.0 + before - after
        raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def enumerate_states(self):
        discovered = {self.start}
        queue = deque([self.start])
        while queue:
            state = queue.popleft()
            if state[1] >= MAX_GUESSES:
                continue
            for action in self.actions:
                for _, next_state, _, _, _ in self.transitions(state, action):
                    if next_state is not None and next_state not in discovered:
                        discovered.add(next_state)
                        queue.append(next_state)
        return tuple(sorted(discovered, key=lambda s: (s[1], s[0])))

    def value_iteration(self, tolerance=1e-12, max_iterations=1000):
        values = {state: 0.0 for state in self.states}
        policy = {}
        for _ in range(max_iterations):
            delta = 0.0
            updated = {}
            for state in reversed(self.states):
                if state[1] >= MAX_GUESSES:
                    updated[state] = 0.0
                    continue
                best_value, best_action = -math.inf, None
                for action in self.actions:
                    q = sum(p * (r + (values[nxt] if nxt is not None else 0.0))
                            for p, nxt, r, _, _ in self.transitions(state, action))
                    if q > best_value:
                        best_value, best_action = q, action
                updated[state] = best_value
                policy[state] = best_action
                delta = max(delta, abs(best_value - values[state]))
            values = updated
            if delta < tolerance:
                break
        return values, policy

    def policy_iteration(self, tolerance=1e-12, max_iterations=1000):
        policy = {s: self.actions[0] for s in self.states if s[1] < MAX_GUESSES}
        values = {state: 0.0 for state in self.states}
        for _ in range(max_iterations):
            # Exact backward evaluation; transitions always advance the turn
            for state in reversed(self.states):
                action = policy.get(state)
                if action is None:
                    values[state] = 0.0
                    continue
                values[state] = sum(
                    p * (r + (values[nxt] if nxt is not None else 0.0))
                    for p, nxt, r, _, _ in self.transitions(state, action)
                )
            stable = True
            for state in self.states:
                if state[1] >= MAX_GUESSES:
                    continue
                best_action = max(
                    self.actions,
                    key=lambda a: sum(
                        p * (r + (values[nxt] if nxt is not None else 0.0))
                        for p, nxt, r, _, _ in self.transitions(state, a)
                    ),
                )
                if policy[state] != best_action:
                    policy[state] = best_action
                    stable = False
            if stable:
                break
        return values, policy

    def expected_guesses(self, policy):
        """Expected guesses to win or reach the six-guess limit."""
        memo = {}

        def evaluate(state):
            if state[1] >= MAX_GUESSES:
                return 0.0
            if state in memo:
                return memo[state]
            total = 0.0
            for p, nxt, _, _, _ in self.transitions(state, policy[state]):
                total += p * (1.0 + (evaluate(nxt) if nxt is not None else 0.0))
            memo[state] = total
            return total

        return evaluate(self.start)


class QLearner:
    def __init__(self, mdp, seed=0, epsilon=0.2, min_epsilon=0.02):
        self.mdp = mdp
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.q = defaultdict(float)
        self.visits = defaultdict(int)

    def choose(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(self.mdp.actions)
        best = max(self.q[(state, a)] for a in self.mdp.actions)
        return self.rng.choice([a for a in self.mdp.actions
                                if self.q[(state, a)] == best])

    def train(self, episodes=100_000, report_every=1000, eval_episodes=1000):
        reports = []
        for episode in range(1, episodes + 1):
            secret = self.rng.choice(self.mdp.words)
            candidates = self.mdp.words
            turn = 0
            epsilon = max(self.min_epsilon,
                          self.epsilon * (1.0 - episode / episodes))
            while turn < MAX_GUESSES:
                state = (candidates, turn)
                action = self.choose(state, epsilon)
                pattern = feedback(action, secret)
                won = pattern == ALL_GREEN
                next_candidates = tuple(w for w in candidates
                                        if feedback(action, w) == pattern)
                done = won or turn + 1 >= MAX_GUESSES
                next_state = None if done else (next_candidates, turn + 1)
                reward = self.mdp.reward(state, action, next_candidates,
                                         won, done)
                key = (state, action)
                self.visits[key] += 1
                alpha = 1.0 / (1.0 + 0.05 * self.visits[key])
                future = (max(self.q[(next_state, a)] for a in self.mdp.actions)
                          if next_state is not None else 0.0)
                self.q[key] += alpha * (reward + future - self.q[key])
                if done:
                    break
                candidates, turn = next_candidates, turn + 1
            if episode % report_every == 0 or episode == episodes:
                reports.append((episode, self.evaluate(eval_episodes)))
        return reports

    def evaluate(self, episodes=1000):
        total_guesses = 0
        for _ in range(episodes):
            secret = self.rng.choice(self.mdp.words)
            candidates, turn = self.mdp.words, 0
            while turn < MAX_GUESSES:
                action = self.choose((candidates, turn), 0.0)
                pattern = feedback(action, secret)
                total_guesses += 1
                if pattern == ALL_GREEN:
                    break
                candidates = tuple(w for w in candidates
                                   if feedback(action, w) == pattern)
                turn += 1
        return total_guesses / episodes