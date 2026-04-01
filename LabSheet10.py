# Import libraries
import gymnasium as gym
import numpy as np
import random
import pandas as pd

# Implement Frozen Lake Environment

# Create environment
env = gym.make("FrozenLake-v1", render_mode="human")

# Initialize Q-table
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0

# Epsilon-greedy policy
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(range(env.action_space.n), key=lambda x: Q[(state, x)])

# Hyperparameters
alpha = 0.85
gamma = 0.90
epsilon = 0.8

# Training parameters
num_eps = 500
num_steps = 100

# Training loop (Q-learning)
for i in range(num_eps):
    state, _ = env.reset()

    for t in range(num_steps):
        action = epsilon_greedy(state, epsilon)

        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning target (OFF-POLICY: max over next actions)
        best_next_action = np.argmax(
            [Q[(next_state, a)] for a in range(env.action_space.n)]
        )

        Q[(state, action)] += alpha * (
            reward + gamma * Q[(next_state, best_next_action)] - Q[(state, action)]
        )

        state = next_state

        if done or truncated:
            break

# Convert Q-table into DataFrame
q_table_df = pd.DataFrame(
    [((s, a), Q[(s, a)]) for (s, a) in Q],
    columns=["(State, Action)", "Q-Value"]
)

print("\nQ-Table:")
print(q_table_df)

# Extract optimal policy
policy = {}
for s in range(env.observation_space.n):
    best_action = max(range(env.action_space.n), key=lambda a: Q[(s, a)])
    policy[s] = best_action

print("\nOptimal Policy (State -> Action):")
for state, action in policy.items():
    print(f"{state} -> {action}")