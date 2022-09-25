#!/usr/bin/env python
# coding: utf-8

# In[58]:


from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
data = load_iris()
print(data.DESCR)


# In[59]:


from sklearn.preprocessing import label_binarize
x = pd.DataFrame(data['data'])
#data的feature_names指定成x的column名稱
x.columns = data['feature_names']
y = data['target']
targets = data['target_names']
print(targets)


# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)


# In[53]:


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test,prediction)
print(cm)


# In[54]:


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
 
color = 'black'
matrix = plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.show()


# In[57]:


report = classification_report(y_test, prediction)
print(report)


# In[ ]:





# In[ ]:




