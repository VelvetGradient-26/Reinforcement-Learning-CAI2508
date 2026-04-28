import numpy as np

# LabSheet 11: Epsilon Greedy Method to find the best arm in MAB

# Custom NumPy Environment replacing gym_bandits
class BanditTwoArmedHighLowFixed:
    def __init__(self):
        # Inner class to mimic the gym env.action_space interface
        class ActionSpace:
            def __init__(self, n):
                self.n = n
            def sample(self):
                return np.random.randint(0, self.n)
        
        self.action_space = ActionSpace(2)
        
        # High (80%) and Low (20%) fixed payout probabilities
        self.p_dist = [0.8, 0.2] 

    def reset(self):
        # Bandits are stateless, so the state is just 0
        return 0

    def step(self, action):
        # Generate a reward of 1 or 0 based on the selected arm's probability
        if np.random.uniform(0, 1) < self.p_dist[action]:
            reward = 1
        else:
            reward = 0
            
        # Returns: next_state, reward, done, info
        return 0, reward, True, {}

# --- Your LabSheet 11 Code ---

env = BanditTwoArmedHighLowFixed()

print("Number of arms:", env.action_space.n)
print("Probability distribution:", env.p_dist)

count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

def epsilon_greedy(epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q)

env.reset()

for i in range(num_rounds):
    arm = epsilon_greedy(epsilon=0.5)
    next_state, reward, done, info = env.step(arm)
    
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print("\nAction counts:", count)
print("Q-values:", Q)
print('The optimal arm is arm {}'.format(np.argmax(Q) + 1))