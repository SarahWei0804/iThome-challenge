#!/usr/bin/env python
# coding: utf-8

# In[32]:


import tensorflow as tf
#SGD
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
var = tf.Variable(1.0)
loss = lambda: (var ** 2)        # d(loss)/d(var1) = var1
step_count = opt.minimize(loss, [var]).numpy()
# Step is `- learning_rate * grad`
print(var.numpy())


# In[37]:


##momentum
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
var = tf.Variable(1.0)
val0 = var.value()
loss = lambda: (var ** 2)         # d(loss)/d(var1) = var1
# First step is `- learning_rate * grad`
step_count = opt.minimize(loss, [var]).numpy()
val1 = var.value()
print(val1.numpy())

# On later steps, step-size increases because of momentum
step_count = opt.minimize(loss, [var]).numpy()
val2 = var.value()
print(val2.numpy())


# In[34]:


opt = tf.keras.optimizers.Adagrad(learning_rate=0.1)
var1 = tf.Variable(1.0)
loss = lambda: (var1 ** 2)       # d(loss)/d(var1) == var1
step_count = opt.minimize(loss, [var1]).numpy()
# The first step is `-learning_rate*sign(grad)`
var1.numpy()


# In[35]:


opt = tf.keras.optimizers.Adam(learning_rate=0.1)
var1 = tf.Variable(1.0)
loss = lambda: (var1 ** 2)       # d(loss)/d(var1) == var1
step_count = opt.minimize(loss, [var1]).numpy()
# The first step is `-learning_rate*sign(grad)`
var1.numpy()


# In[36]:


## RMSprop
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
var1 = tf.Variable(1.0)
loss = lambda: (var1 ** 2)
step_count = opt.minimize(loss, [var1]).numpy()
var1.numpy()


# In[ ]:




