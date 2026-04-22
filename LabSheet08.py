import gymnasium as gym
import pandas as pd

# create the environment
env = gym.make("FrozenLake-v1",render_mode ="human")

env.reset()

# define a random input policy
def random_policy():
    return env.action_space.sample()

# initialize the value of all states to zeros
V = {}
for s in range(env.observation_space.n):
    V[s] = 0.0

# initialize the parameters
alpha = 0.85
gamma = 0.90
num_eps = 100
num_steps = 50

# generating episodes
for i in range(num_eps):
    s = env.reset()
    s = s[0]

    for t in range(num_steps):
        a = random_policy()
        s_, r, done, _, _ = env.step(a)
        # use TD-update rule
        V[s] += alpha * (r + gamma * V[s_] - V[s])
        s = s_
        if done:
            break

# convert the dictionary to a data frame
df = pd.DataFrame(list(V.items()), columns = ['state','value'])
print(df)