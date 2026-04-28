import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# LabSheet 15: Banner Advertisement

# Create a synthetic dataset representing click-throughs for 5 banners
df = pd.DataFrame()
for i in range(5):
    df['Banner_type_' + str(i)] = np.random.randint(0, 2, 100000)

num_iterations = 10000
num_banner = 5
count = np.zeros(num_banner)
sum_rewards = np.zeros(num_banner)
Q = np.zeros(num_banner)
banner_selected = [] 

def epsilon_greedy_policy(epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_banner)
    else:
        return np.argmax(Q)

for i in range(num_iterations):
    banner = epsilon_greedy_policy(0.5)
    # Simulate displaying the banner and getting a reward (click = 1, no click = 0)
    reward = df.values[i, banner]
    
    count[banner] += 1
    sum_rewards[banner] += reward
    Q[banner] = sum_rewards[banner] / count[banner]
    banner_selected.append(banner)

print('The best banner is banner {}'.format(np.argmax(Q) + 1))

# Plot the distribution of selected banners
ax = sns.countplot(x=banner_selected)
ax.set(xlabel='Banner', ylabel='Count')
plt.show()