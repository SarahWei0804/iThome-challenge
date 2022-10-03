#!/usr/bin/env python
# coding: utf-8

# In[29]:


import matplotlib.pyplot as plt
def plot(model, attr, width, height):
    plt.figure(figsize=(width, height))
    plt.plot(model.history[attr], color = 'r', label = attr)
    plt.plot(model.history['val_'+attr], color = 'b', label = 'val_'+attr)
    plt.xlabel('Epochs')
    plt.ylabel(attr)
    plt.legend()
    plt.show()


# In[24]:


import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 5, validation_split = 0.2)

model.evaluate(x_test, y_test)


# In[37]:


f1 = plot(history, 'accuracy', 5, 3)
f2 = plot(history, 'loss', 5, 3)


# In[31]:


import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

Input = tf.keras.layers.Input(shape=(28,28))
layer1 = tf.keras.layers.Flatten()(Input)
layer2 = tf.keras.layers.Dense(128, activation = 'relu')(layer1)
layer3 = tf.keras.layers.Dropout(0.2)(layer2)
Output = tf.keras.layers.Dense(10, activation = 'softmax')(layer3)
model2 = tf.keras.Model(inputs = Input, outputs = Output)
model2.summary()
model2.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

history2 = model2.fit(x_train, y_train, epochs = 5, validation_split = 0.2)

model2.evaluate(x_test, y_test)


# In[38]:


plot(history2, 'accuracy', 5, 3)
plot(history2, 'loss', 5, 3)


# In[ ]:




