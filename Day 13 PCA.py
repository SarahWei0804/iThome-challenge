#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()
x = pd.DataFrame(data['data'])
x.columns = data['feature_names']
y = pd.DataFrame(data['target']).values


# In[21]:


n_components = 2
pca = PCA(n_components = n_components)
pca = pca.fit(x)

transformed = pca.transform(x)


# In[23]:


var_exp=pca.explained_variance_ratio_ #獲得貢獻率
np.set_printoptions(suppress=True) #當suppress=True，表示小數不需要以科學計數法的形式輸出
print('各主成分貢獻率：',var_exp)

cum_var_exp=np.cumsum(var_exp)  #累計貢獻度
print('各主成分累積貢獻率：',cum_var_exp)


# In[ ]:




