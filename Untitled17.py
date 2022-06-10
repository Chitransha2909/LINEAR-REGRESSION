#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING ALL THE NECESSARY LIBRARIES

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## LOAD THE DATADET

# In[7]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"


# In[8]:


df=pd.read_csv(url)


# In[9]:


df.head()


# In[11]:


df.info()


# ## PLOT THE RELATIONSHIP BETWEEN THE VARIABLES TO SHOW THE RELATIONSHIP OF THE VARIABLES

# In[13]:


df.plot(x="Hours" , y="Scores",style="o")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Relationship between the two variables")
plt.show()


# ## PREPARING THE DATA

# In[21]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# ### DIVIDING THE DATA INTO TWO PARTS

# In[22]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.3, random_state=50) 


# In[23]:


from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(x_train, y_train) 


# In[24]:


reg.coef_


# In[25]:


reg.intercept_


# In[27]:


line = reg.coef_ * x + reg.intercept_


# In[31]:


plt.scatter(x,y)


# In[29]:


plt.plot(x,line)


# ##### Thus, there is a linear relationship between the two variables

# In[30]:


y_pred = reg.predict(x_test)


# #### comparing the scores between the actual and predicted values

# In[32]:


a = pd.DataFrame({"actual":y_test, "predicted" : y_pred})


# In[33]:


a


# ### What will be the score if the student works 9.25hours a day

# In[36]:


hours= 9.25
p=reg.predict([[hours]])
print("No of Hours = {} and it's predicted  Score = {}%".format(h,p))


# In[39]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




