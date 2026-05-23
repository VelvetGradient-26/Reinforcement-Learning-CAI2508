import numpy as np

# Custom Environment
class BanditTwoArmedHighLowFixed: 
    def __init__(self):
        class ActionSpace: # mimicking env.action_space
            def __init__(self, n): 
                self.n = n # n = number of arms for the bandit

            def sample(self): 
                return np.random.randint(0, self.n) # mimicking env.action_space.sample()
            
        # create ActionSpace Object
        self.action_space = ActionSpace(2) # modify this to add more arms to the bandit

        # create the probability distribution manually
        self.p_dist = [0.8, 0.2] # arm 0 and arm 1 (agent doesn't know that arm o is the best)

    # bandit problems are stateless (no env movement like FrozenLake)
    def reset(self): 
        return 0
    
    # step function (mimicking env.step())
    def step(self, action): 
        if np.random.uniform(0, 1) < self.p_dist[action]: 
            reward = 1
        else: 
            reward = 0

        # next_state, reward, done, info        
        return 0, reward, True, {} # bandit has no real states that's why zero
    
env = BanditTwoArmedHighLowFixed()

print(env.action_space.n)
print(env.p_dist)

# EPSILON GREEDY POLICY
count = np.zeros(2) # how many times each arm is selected
sum_rewards = np.zeros(2) # tracks total reward from each sum
Q = np.zeros(2) # agent knows nothing initially
num_rounds = 100 # Algorithm runs _ times

def epsilon_greedy(epsilon): 
    if np.random.uniform(0,1) < epsilon: # explore (random arm selection)
        return env.action_space.sample()
    else:                               # exploit (choose arm with highest q-value)
        return np.argmax(Q)
    
env.reset()

for i in range(num_rounds): 
    arm = epsilon_greedy(epsilon=0.5) # tweak epsilon value here
    next_state, reward, done, info = env.step(arm)

    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print("Epsilon Greedy Policy: ")
print("Action counts: ", count)
print("Q-values: ", Q)
print(f"Optimal arm is: {np.argmax(Q) + 1}\n")


# SOFTMAX EXPLORATION
count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

def softmax(T): 
    # manual softmax implementation 
    denom = sum([np.exp(i/T) for i in Q])
    probs = [np.exp(i/T)/denom for i in Q]

    # or use scipy
    from scipy.special import softmax
    probs = softmax(Q/T)

    # choose an arm based on softmax probablities
    arm = np.random.choice(env.action_space.n, p=probs)
    return arm

env.reset()

for i in range(num_rounds): 
    arm = softmax(T=50) # tweak temperature value here
    next_state, reward, done, info = env.step(arm)

    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print("Softmax Exploration: ")
print("Action counts: ", count)
print("Q-values: ", Q)
print(f"Optimal arm is: {np.argmax(Q) + 1}\n")


# UPPER CONFIDENCE BOUND ALGORITHM
count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

def UCB(i):
    # store ucb values for each arm
    ucb = np.zeros(2)

    if i < 2: # prevent division by zero crash
        return i 
    else: # formula implementation
        for arm in range(2): 
            ucb[arm] = Q[arm] + np.sqrt((2 * np.log(sum(count))) / count[arm])
    
    return np.argmax(ucb)

env.reset()
for i in range(num_rounds): 
    arm = UCB(i)
    next_state, reward, done, info = env.step(arm)

    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print("UCB Algorithm: ")
print("Action counts: ", count)
print("Q-values: ", Q)
print(f"Optimal arm is: {np.argmax(Q) + 1}\n")

# THOMPSON SAMPLING
count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

alpha = np.ones(2)
beta = np.ones(2)

def thomson_sampling(alpha, beta): 
    samples = []
    for i in range(2): 
        # generate random samples from beta distribution
        sample = np.random.beta(alpha[i] + 1, beta[i] + 1)
        samples.append(sample)

    return np.argmax(samples)

env.reset()

for i in range(num_rounds): 
    arm = thomson_sampling(alpha, beta)
    next_state, reward, done, info = env.step(arm)

    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

    # belief update
    if reward == 1:  
        alpha[arm] += 1
    else: 
        beta[arm] += 1

print("Thompson Sampling: ")
print("Action counts: ", count)
print("Q-values: ", Q)
print(f"Optimal arm is: {np.argmax(Q) + 1}\n")