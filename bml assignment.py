#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


data=[[43,99],[21,65],[25,79],[42,74],[57,87],[59,81]]


# In[3]:


ds=pd.DataFrame(data, columns=['AGE','GLUCOSE LEVEL'])
ds.set_index(np.array(range(1,7)))


# In[4]:


x=np.array(ds['AGE'])
y=np.array(ds['GLUCOSE LEVEL'])
print(y)
print(x)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


plt.scatter(x,y)
plt.xlabel('AGE')
plt.ylabel('GLUCOSE LEVEL')


# In[7]:


#object for regression
x=np.array(x.reshape((-1, 1)))
reg = linear_model.LinearRegression()
reg.fit(x,y)


# In[8]:


reg.predict(np.array([55]).reshape(-1, 1))


# In[9]:


reg.coef_ 


# In[10]:


reg.intercept_


# In[ ]:




