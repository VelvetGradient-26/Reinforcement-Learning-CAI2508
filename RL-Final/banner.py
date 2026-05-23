import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# create a synthetic dataset representing clicks and banners
df = pd.DataFrame()
for i in range(5): 
    df['Banner_type' + str(i)] = np.random.randint(0, 2, 100000)

df.to_csv('/Users/deepak/Desktop/Practice/RL-Final/df.csv')

iterations = 10000
banners = 5
count = np.zeros(banners)
sum_rewards = np.zeros(banners)
Q = np.zeros(banners)
banner_selected = []

def epsilon_greedy(epsilon): 
    if np.random.uniform(0, 1) < epsilon: 
        return np.random.choice(banners)
    else: 
        return np.argmax(Q)
    
for i in range(iterations): 
    banner = epsilon_greedy(0.5)
    reward = df.values[i, banner]

    count[banner] += 1
    sum_rewards[banner] += 1
    Q[banner] = sum_rewards[banner]/count[banner]
    banner_selected.append(banner)

print(f"The best banner is: {np.argmax(Q) + 1}")
    
# Plot the distribution of selected banners
ax = sns.countplot(x=banner_selected)
ax.set(xlabel='Banner', ylabel='Count')
plt.show()