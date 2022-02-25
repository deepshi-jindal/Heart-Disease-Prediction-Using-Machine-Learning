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


heart.shape


# In[6]:


heart.tail()


# In[7]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(heart.columns[i]+":\t\t\t"+info[i])


# In[8]:


heart.info()


# In[9]:


heart.isnull().sum()


# In[10]:


heart.describe()


# In[11]:


heart['target'].value_counts()


# In[12]:


heart['target'].isnull()


# In[13]:


heart['target'].sum()


# In[14]:


heart['target'].unique()


# In[15]:


X=heart.drop(columns='target',axis=1)
Y=heart['target']


# In[16]:


X


# In[17]:


Y


# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[19]:


print(X.shape,X_train.shape,X_test.shape)


# In[20]:


print(Y.shape,Y_train.shape,Y_test.shape)


# In[21]:


X_test


# In[22]:


Y_test


# In[23]:


Z_model=LogisticRegression()


# In[24]:


Z_model.fit(X_train,Y_train)


# In[25]:


X_train_prediction=Z_model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[26]:


print('Accuracy on Training :',training_data_accuracy)
print('Percentage accuracy on Training:',(training_data_accuracy)*100)


# In[27]:


X_test_prediction=Z_model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[28]:


print('Accuracy on Test data:',test_data_accuracy)
print('Percentage accuracy on Test data:',(test_data_accuracy)*100)


# In[29]:


input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=Z_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The Person does not have heart disease")
else:
    print('The person has heart Disease')


# In[ ]:





# In[ ]:




