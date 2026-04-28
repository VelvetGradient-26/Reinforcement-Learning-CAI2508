import numpy as np

# Custom NumPy Environment replacing gym_bandits
class BanditTwoArmedHighLowFixed:
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

# Labsheet 12: Softmax Exploration to find the best arm in MAB

env = BanditTwoArmedHighLowFixed()

count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

def softmax(T):
    # Calculate the denominator for the softmax probabilities
    denom = sum([np.exp(i/T) for i in Q])
    # Calculate the probability for each arm
    probs = [np.exp(i/T)/denom for i in Q]
    # Choose an arm based on the softmax probabilities
    arm = np.random.choice(env.action_space.n, p=probs)
    return arm

env.reset()

# Initial Temperature
T = 50 

for i in range(num_rounds):
    arm = softmax(T)
    next_state, reward, done, info = env.step(arm)
    
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]
    
    # Temperature annealing: gradually reduce T to shift from exploration to exploitation
    T = T * 0.99

print("Action counts:", count)
print("Q-values:", Q)
print('The optimal arm is arm {}'.format(np.argmax(Q) + 1))