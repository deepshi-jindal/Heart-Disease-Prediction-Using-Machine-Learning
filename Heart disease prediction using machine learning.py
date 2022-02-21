#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


heart=pd.read_csv('heart.csv')


# In[3]:


heart


# In[4]:


heart.head()


# In[5]:


heart.tail()


# In[6]:


heart.shape


# In[7]:


heart.info()


# In[8]:


heart.isnull().sum()


# In[9]:


heart.describe()


# In[10]:


heart['target'].value_counts()


# In[11]:


X=heart.drop(columns='target',axis=1)
Y=heart['target']


# In[12]:


X


# In[13]:


Y


# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[15]:


print(X.shape,X_train.shape,X_test.shape)


# In[16]:


print(Y.shape,Y_train.shape,Y_test.shape)


# In[19]:


Z=LogisticRegression()


# In[20]:


Z.fit(X_train,Y_train)


# In[21]:


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[27]:


print('Accuracy on Training :',training_data_accuracy)
print('Percentage accuracy on Training:',(training_data_accuracy)*100)


# In[23]:


X_test_prediction=Z.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[28]:


print('Accuracy on Test data:',test_data_accuracy)
print('Percentage accuracy on Test data:',(test_data_accuracy)*100)


# In[29]:


input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=Z.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The Person does not have heart disease")
else:
    print('The person has heart Disease')


# In[ ]:




