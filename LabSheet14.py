import numpy as np

# LabSheet 14: Thompson Sampling

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
        reward = 1 if np.random.uniform(0, 1) < self.p_dist[action] else 0
        return 0, reward, True, {}
    
# (Assuming the same BanditTwoArmedHighLowFixed class is defined above)
env = BanditTwoArmedHighLowFixed()

count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)

# Alpha tracks successes, Beta tracks failures
alpha = np.ones(2)
beta = np.ones(2)
num_rounds = 100

def thompson_sampling(alpha, beta):
    # Sample from the Beta distribution for each arm
    samples = [np.random.beta(alpha[i] + 1, beta[i] + 1) for i in range(2)]
    return np.argmax(samples)

env.reset()

for i in range(num_rounds):
    arm = thompson_sampling(alpha, beta)
    next_state, reward, done, info = env.step(arm)
    
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]
    
    # Update alpha or beta based on the reward outcome
    if reward == 1:
        alpha[arm] = alpha[arm] + 1
    else:
        beta[arm] = beta[arm] + 1

print("Q-values:", Q)
print('The optimal arm is arm {}'.format(np.argmax(Q) + 1))
    
