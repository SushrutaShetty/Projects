#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
df = pd.read_csv('headbrain.csv', sep =',')
df


# In[7]:


x = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values


# In[10]:


mean_x = np.mean(x)
mean_y = np.mean(y)

m = len(x)

numerator = 0
denominator = 0
for i in range(m):
    numerator += (x[i] - mean_x) * (y[i] - mean_y)
    denominator += (x[i] - mean_x) ** 2
slope = numerator / denominator
c = mean_y - (slope * mean_x)

print(slope, c)


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,10)

max_x = np.max(x) + 100
min_x = np.min(x) - 100

X = np.linspace(min_x, max_x, 1000)
Y = slope * X + c

plt.plot(X, Y, color = "#58b970", label = 'Regression Line')
plt.scatter(x,y, color = '#ef5423', label = 'Scatter Plot')

plt.xlabel('Head Size')
plt.ylabel('Brain Size')
plt.legend()
plt.show()


# In[15]:


total_sos = 0
total_sores = 0
for i in range(m):
    y_pred = c+ slope*x[i]
    total_sos = (y[i] - mean_y) ** 2
    total_sores = (y[i] - y_pred) ** 2
r_squared = 1 - (total_sores/total_sos)
print(r_squared)

