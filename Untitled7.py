#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pdy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


heart=pd.read_csv('heart.csv')


# In[7]:


heart


# In[8]:


heart.head()


# In[9]:


heart.tail()


# In[10]:


heart.shape


# In[11]:


heart.info()


# In[12]:


heart.isnull().sum()


# In[13]:


heart.describe()


# In[15]:


heart['target'].value_counts()


# In[16]:


X=heart.drop(columns='target',axis=1)
Y=heart['target']


# In[17]:


print(X)


# In[18]:


print(Y)


# In[20]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[21]:


print(X.shape,X_train.shape,X_test.shape)


# In[22]:


print(Y.shape,Y_train.shape,Y_test.shape)


# In[23]:


model=LogisticRegression()


# In[24]:


model.fit(X_train,Y_train)


# In[25]:


X_train_prediction=Z.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[26]:


print('Accuracy on Training :',training_data_accuracy)


# In[31]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[32]:


print('Accuracy on Test data:',test_data_accuracy)


# In[6]:



input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0.2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
print()
prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The Persin does not have heart disease")
else:
    print('The person has heart Disease')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




