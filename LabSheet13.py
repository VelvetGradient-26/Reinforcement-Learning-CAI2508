import numpy as np

# LabSheet 13: Upper Confidence Bound Algorithm

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

# UCB Implementation 
env = BanditTwoArmedHighLowFixed()

count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

def UCB(i):
    ucb = np.zeros(2)
    # Explore each arm at least once initially
    if i < 2:
        return i
    else:
        for arm in range(2):
            # Calculate the UCB value for each arm
            ucb[arm] = Q[arm] + np.sqrt((2 * np.log(sum(count))) / count[arm])
        return np.argmax(ucb)

env.reset()

for i in range(num_rounds):
    arm = UCB(i)
    next_state, reward, done, info = env.step(arm)
    
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print("Q-values:", Q)
print('The optimal arm is arm {}'.format(np.argmax(Q) + 1))