import gymnasium as gym

frozen_lake = gym.make("FrozenLake-v1", render_mode='human')
cartpole = gym.make("CartPole-v1", render_mode='human')

def main(env, episodes, timesteps):
    print(env.observation_space)
    print(env.action_space)
    for episode in range(episodes): 
        rewards = []
        env.reset()
        for timestep in range(timesteps): 
            random_action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(random_action)
            rewards.append(reward)
            env.render()
            done = terminated or truncated
            if done: 
                break 
        print(f"Return for episode {episode + 1}: {sum(rewards)}")

# pass env, number of episodes and number of timesteps
main(frozen_lake, 10, 10)
    