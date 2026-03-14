import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

print("Action Space:")
print(env.action_space)

print("State Space:")
print(env.observation_space)

state, _ = env.reset()

random_action = env.action_space.sample()

next_state, reward, done, truncated, info = env.step(random_action)

# e) Generate episodes using a random policy
episodes = 5
timesteps = 20

print("Running Random Policy Episodes\n")

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    for t in range(timesteps):
        env.render()
        # random policy
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if done or truncated:
            break

    print(f"Episode {episode+1} Return: {total_reward}")

env.close()