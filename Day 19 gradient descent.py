#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def func(x): return x ** 2

def dfunc(x): return 2*x

def GD(start, df, epochs, lr):
    xs = np.zeros(epochs+1)
    x = start
    xs[0]=x
    for i in range(epochs):
        x += - lr*df(x)
        xs[i+1] = x
    return xs


# In[3]:


start = 5
epochs = 15
lr = 0.3
w = GD(start, dfunc, epochs, lr)
print(np.around(w,2))


# In[7]:


t = np.arange(-6,6,0.01)
plt.plot(t, func(t), c = 'b')
plt.plot(w, func(w), c = 'r', label = 'lr = {}'.format(lr))
plt.scatter(w, func(w), c = 'r')
plt.title('Gradient descent')
plt.xlabel('X')
plt.ylabel('loss function')
plt.show()


# In[ ]:




