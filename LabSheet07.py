# Experiment 7:
# Implement On-Policy Monte Carlo Control with Epsilon-Greedy Policy
# for Blackjack environment

import gymnasium as gym
import random
from collections import defaultdict
import pandas as pd

# Create Blackjack environment
env = gym.make("Blackjack-v1", render_mode="human")

# Initialize data structures
Q = defaultdict(float)              # Action-value function
total_return = defaultdict(float)   # Sum of returns
N = defaultdict(int)                # Count of visits


def epsilon_greedy_policy(state, Q, epsilon=0.5):
    """
    Select action using epsilon-greedy strategy.
    """
    if random.uniform(0, 1) < epsilon:
        # Explore: random action
        return env.action_space.sample()
    else:
        # Exploit: best action based on Q-values
        return max(
            list(range(env.action_space.n)),
            key=lambda x: Q[(state, x)]
        )


def generate_episode(Q, num_timesteps=100):
    """
    Generate an episode using the current policy.
    """
    episode = []
    state, _ = env.reset()

    for t in range(num_timesteps):
        action = epsilon_greedy_policy(state, Q)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        if terminated or truncated:
            break

        state = next_state

    return episode


# Number of training iterations
num_iterations = 100

for i in range(num_iterations):
    # Generate episode
    episode = generate_episode(Q)

    # Extract state-action pairs
    all_state_action_pairs = [(s, a) for (s, a, r) in episode]

    # Extract rewards
    rewards = [r for (s, a, r) in episode]

    # Loop through episode
    for t, (state, action, _) in enumerate(episode):

        # First-visit MC check
        if (state, action) not in all_state_action_pairs[:t]:

            # Compute return from time t
            R = sum(rewards[t:])

            # Update total return and count
            total_return[(state, action)] += R
            N[(state, action)] += 1

            # Update Q value
            Q[(state, action)] = (
                total_return[(state, action)] / N[(state, action)]
            )


# Convert Q-table to DataFrame for display
df = pd.DataFrame(Q.items(), columns=["State-Action Pair", "Value"])

print(df.head(11))
print(df.iloc[124:130])


# Test the learned policy
for ep in range(10):
    state, _ = env.reset()
    done = False

    while not done:
        action = epsilon_greedy_policy(state, Q, epsilon=0.1)

        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state

        if terminated or truncated:
            done = True

# Close environment
env.close()