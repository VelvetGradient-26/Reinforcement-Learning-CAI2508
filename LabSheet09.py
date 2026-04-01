# Import libraries
import gymnasium as gym
import random
import pandas as pd

# Implement SARSA in frozen lake environment

# Create the environment
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
alpha = 0.85      # learning rate
gamma = 0.90      # discount factor
epsilon = 0.8     # exploration rate

# Training parameters
num_eps = 500
num_steps = 100

# Training loop
for i in range(num_eps):
    state, _ = env.reset()
    action = epsilon_greedy(state, epsilon)

    for t in range(num_steps):
        next_state, reward, done, truncated, _ = env.step(action)

        next_action = epsilon_greedy(next_state, epsilon)

        # Q-learning update
        predict = Q[(state, action)]
        target = reward + gamma * Q[(next_state, next_action)]
        Q[(state, action)] += alpha * (target - predict)

        state = next_state
        action = next_action

        if done or truncated:
            break

# Convert Q-table into a DataFrame
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