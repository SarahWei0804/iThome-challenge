#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
columns = ['variance of Wavelet Transformed image', 'skewness of Wavelet Transformed image', 'curtosis of Wavelet Transformed image', 'entropy of image', 'target']
data = pd.read_csv(url, names = columns)
data


# In[3]:


data[data['target'] == 0].describe()


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
x = data.iloc[:,:-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# In[18]:


svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)
prediction = svm.predict(x_test)


# In[22]:


from sklearn.metrics import plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
color = 'black'
cm = plot_confusion_matrix(svm, x_test, y_test, cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.xlabel("Predicted value", color = color)
plt.ylabel("Actual value", color = color)
plt.show()


# In[24]:


cm = confusion_matrix(y_test, prediction)
cm


# In[21]:


report = classification_report(y_test, prediction)
print(report)


# In[26]:


from sklearn.metrics import roc_curve, roc_auc_score, auc

# 在各種『決策門檻』（decision threshold）下，計算 『真陽率』（True Positive Rate；TPR）與『假陽率』（False Positive Rate；FPR）
fpr, tpr, threshold = roc_curve(y_test, prediction)

auc = auc(fpr, tpr)
## Plot the result
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()    


# In[ ]:




