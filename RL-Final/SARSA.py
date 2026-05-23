import pandas as pd
import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1')

# Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Epsilon greedy policy
def epsilon_greedy(state, epsilon): 
    if np.random.uniform(0, 1) < epsilon: 
        return env.action_space.sample()
    return np.argmax(Q[state])

# hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# training settings
episodes = 1000000
timesteps = 100

for episode in range(episodes): 
    state, _ = env.reset()
    action = epsilon_greedy(state, epsilon)
    for step in range(timesteps): 
        next_state, reward, terminated, truncated, info = env.step(action)

        # SARSA update
        next_action = epsilon_greedy(state, epsilon)
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        # # Q-learing update
        # best_next_action = np.argmax(Q[next_state])
        # Q[state, action] += alpha * (
        #     reward + gamma * Q[next_state, best_next_action] - Q[state, action]
        # )


        state = next_state
        action = next_action

        if terminated or truncated: 
            break

# Convert Q-table to dataframe
q_table_df = pd.DataFrame(Q)

print("Q-Table: ")
print(q_table_df)

# optimal policy
policy = np.argmax(Q, axis=1)

print("Optimal Policy: ")
for state in range(env.observation_space.n): 
    print(f"{state} -> {policy[state]}")
