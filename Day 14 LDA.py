#!/usr/bin/env python
# coding: utf-8

# In[23]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
## load the iris dataset
data = load_iris()


# In[24]:


## extract the variables
x = pd.DataFrame(data['data'])
x.columns = data['feature_names']
y = data['target']


# In[25]:


## the dimension of x is 4, and the dimension of y is 3. Therefore, we only can reduce the dimension to 2 (3-1).
lda = LinearDiscriminantAnalysis(n_components = 2)
lda_x = lda.fit_transform(x, y)


# In[26]:


## Visualize the transformed data
import matplotlib.pyplot as plt

plt.scatter(lda_x[:,0], lda_x[:,1], c = y)


# In[27]:


## 取得2個變數的貢獻率並且畫圖
ratio = lda.explained_variance_ratio_
np.set_printoptions(suppress = True)
print('各變數貢獻率: ', ratio)
plt.bar(['Variable 1' ,'Variable 2'], ratio)

