import gymnasium as gym
import pandas as pd

env = gym.make("FrozenLake-v1", render_mode='human')

# define a random input policy
def random_policy(): 
    return env.action_space.sample()

# init the value of all states to zeros
V = {}
for state in range(env.observation_space.n): 
    V[state] = 0.0

# initilize the parameters
alpha = 0.85
gamma = 0.90
episodes = 100
timesteps = 50

for episode in range(episodes): 
    state = env.reset()
    state = state[0]

    for timestep in range(timesteps): 
        action = random_policy()
        next_state, reward, terminated, truncated, info = env.step(action)

        # Use TD-update rule
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state
        if terminated: 
            break

# convert the dictionary to a dataframe
df = pd.DataFrame(list(V.items()), columns=['state', 'value'])
print(df)