#!/usr/bin/env python
# coding: utf-8

# In[78]:


from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
print(data)


# In[73]:


x = pd.DataFrame(data['data'])
x.columns = data['feature_names']
y = pd.DataFrame(data['target'])
y.columns = data['target_names']
print(x.describe(),"\n", y.describe())


# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
regression = LinearRegression()
regression.fit(x_train, y_train)
y_prediction = regression.predict(x_test)


# In[76]:


from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
  
mae = mean_absolute_error(y_test,y_prediction)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_test,y_prediction) #default=True
rmse = mean_squared_error(y_test,y_prediction,squared=False)
r_square = r2_score(y_test, y_prediction)
print("MAE:", round(mae,2), "\nMSE:", round(mse,2), "\nRMSE:", round(rmse,2), "\nR square:", round(r_square,2))


# In[ ]:





# In[ ]:




