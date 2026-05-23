import gymnasium as gym
from collections import defaultdict

# Create Blackjack environment
env = gym.make("Blackjack-v1")

# Simple policy
def policy(state):
    # state[0] = player sum
    if state[0] > 19:
        return 0   # stick
    return 1       # hit


# Generate episode
def generate_episode():
    episode = []
    state, _ = env.reset()

    while True:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, reward))
        state = next_state
        if terminated or truncated:
            break

    return episode


# Store total returns
total_return = defaultdict(float)

# Store visit counts
N = defaultdict(int)
episodes = 10000


# Monte Carlo training
for i in range(episodes):
    episode = generate_episode()
    states, rewards = zip(*episode)
    for t, state in enumerate(states):
        # if state not in states[:t]: first visit check
        # Monte Carlo return
        R = sum(rewards[t:])
        total_return[state] += R
        N[state] += 1


# Print state values
print("\nState Values:\n")

for state in total_return:
    value = total_return[state] / N[state]
    print(state, "->", round(value, 3))