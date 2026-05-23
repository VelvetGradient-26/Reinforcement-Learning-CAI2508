import numpy as np

class BanditTwoArmHighLowFixed: 
    def __init__(self): 
        class ActionSpace: 
            def __init__(self, n): 
                self.n = n
            def sample(self): 
                return np.random.randint(0, self.n)
            
        self.action_space = ActionSpace(2)
        self.p_dist = [0.8, 0.2]

    def reset(self): 
        return 0
    
    def step(self, action): 
        if np.random.uniform(0, 1) < self.p_dist[action]: 
            reward = 1
        else: 
            reward = 0
        return 0, reward, True, {}
    

env = BanditTwoArmHighLowFixed()

count = np.zeros(2)
sum_reward = np.zeros(2)
Q = np.zeros(2)
iterations = 100

def epsilon_greedy(epsilon): 
    if np.random.uniform(0, 1) < epsilon: 
        return env.action_space.sample()
    else: 
        return np.argmax(Q)
    
def softmax(T): 
    denom = sum([np.exp(i/T) for i in Q])
    probs = [np.exp(i/T)/denom for i in Q]

    return np.random.choice(env.action_space.n, p=probs)

env.reset()
for i in range(iterations): 
    arm = epsilon_greedy(epsilon=0.8)
    next_state, reward, terminated, info = env.step(arm)

    count[arm] += 1
    sum_reward[arm] += reward
    Q[arm] = sum_reward[arm] / count[arm]

print("Epsilon Greedy Policy: ")
print("Action count: ", count)
print("Q-values: ", Q)
print("Optimal Arm: ", np.argmax(Q)+1)

env.reset()

sum_rewards = np.zeros(2)
Q = np.zeros(2)
count = np.zeros(2)

for i in range(iterations): 
    arm = softmax(T=50)
    next_state, reward, terminated, info = env.step(arm)

    count[arm] += 1
    sum_reward[arm] += reward
    Q[arm] = sum_reward[arm] / count[arm]

print("Softmax Exploration: ")
print("Action Count: ", count)
print("Q-values: ", Q)
print("Optimal Arm: ", np.argmax(Q)+1)