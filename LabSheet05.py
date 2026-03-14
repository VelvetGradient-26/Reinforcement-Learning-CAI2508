import gymnasium as gym
from collections import defaultdict
import pandas as pd

'''  
Title - Implementation of every visit monte carlo prediction for blackjack environment

'''
env = gym.make('Blackjack-v1', render_mode = "human")
env.reset()
env.render()

def policy(state): 
    return 0 if state[0] > 19 else 1

state = env.reset()
state = state[0]

print(f"State: {state}")
print(f"Policy for the state: {policy(state)}")

num_timesteps = 100

def generate_episode(policy): 
    episode = []
    state = env.reset()
    state = state[0]

    for t in range(num_timesteps): 
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        if terminated or truncated: 
            break

        state = next_state

    return episode

print(f"Sample Episode: {generate_episode(policy)}")

# use defaultdict with default value 0.0 to store total returns 
total_return = defaultdict(float)

# dict to store number of visits for each state
# deafault value = 0
N = defaultdict(int)

num_iterations = 10

# Monte Carlo Policy Evaluation
for i in range(num_iterations): 
    episode = generate_episode(policy)

    states, actions, rewards = zip(*episode)

    for t, state in enumerate(states):
        #compute return
        R = sum(rewards[t:])

        total_return[state] += R

        N[state] += 1

# print total return of the last state processed
print(f"Total return of the last state processed: {total_return[state]}")

# print number of visits of last state
print(N[state])

# Convert total_return dictionary to DataFrame
total_return = pd.DataFrame(
    total_return.items(), 
    columns=['state', 'total_return']
)

# Convert vist_count dictionary to dataframe
N = pd.DataFrame(
    N.items(), 
    columns=['state', 'N']
)

# Merge both DataFrames on state column
df = pd.merge(total_return, N, on='state')

print(df)

# check for specific states
df[df['state'] == (21, 9, 0)]['value'].values
df[df['state'] == (5, 8, 0)]['value'].values

