import numpy as np
import gymnasium as gym

'''   
Solving Frozen Lake using Policy Iteration - Find an Optimal Policy
Aim: <fill aim> 
'''
def compute_value_function(env, policy): 
    num_iterations = 1000
    threshold = 1e-20

    gamma = 1.0

    value_table = np.zeros(env.observation_space.n)

    for i in range(num_iterations): 
        updated_value_table = np.copy(value_table)
        for s in range(env.observation_space.n):
            a = policy[s]
            value_table[s] = sum([prob * (r + gamma * updated_value_table[s_])
                                  for prob, s_, r, _ in env.P[s][a]])
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold): 
            break
    
    return (value_table)


def extract_policy(env, value_table): 
    gamma = 1.0
    # initialize the policy of all states to zero
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n): 
        Q_values = [sum([prob * (r + gamma * value_table[s_])
                                  for prob, s_, r, _ in env.P[s][a]])
                                    for a in range(env.action_space.n)]
        policy[s] = np.argmax(np.array(Q_values))
        
    return policy
    
def policy_iteration(env): 
    num_iterations = 1000
    # initialize the policy of all states to action 0
    policy = np.zeros(env.observation_space.n)
    for i in range(num_iterations): 
        value_function = compute_value_function(env, policy)
        new_policy = extract_policy(env, value_function)

        if(np.all(policy == new_policy)):
            break
        policy = new_policy
    
    return policy

if __name__ == "__main__": 
    env = gym.make('FrozenLake-v1', render_mode="human")
    state, info = env.reset()
    env.render()
    env = env.unwrapped
    optimal_policy = policy_iteration(env)
    print("Optimal Policy:", optimal_policy)

    episodes = 10
    for m in range(episodes): 
        rewards = []
        state, info = env.reset()
        num_timesteps = 50
        for t in range(num_timesteps):
            act = int(optimal_policy[state])
            next_state, reward, terminated, truncated, info = env.step(act)
            rewards.append(reward)
            env.render()
            state = next_state
            if terminated or truncated: 
                break
        print(f"Return for Episode {m+1}: {sum(rewards)}")
