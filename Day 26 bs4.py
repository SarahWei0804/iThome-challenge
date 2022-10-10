#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
from bs4 import BeautifulSoup

url = "https://pets.ettoday.net/"
response = requests.get(url)
bs = BeautifulSoup(response.text, "html.parser")
print(bs)


# In[41]:


result = bs.find("a")
print(result)


# In[42]:


result = bs.find_all("h3")
all_title = []
for i in result:
    try:
        all_title.append(i.find("a").get("title"))
    except:
        continue
print(all_title)


# In[43]:


summary = bs.find_all("p", class_ = "summary", limit = 5)
for i in summary:
    print(i.text)


# In[46]:


print(bs.select(".summary", limit = 3))


# In[ ]:




