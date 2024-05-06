#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[3]:


df = pd.read_csv("./spam.csv")
df.sample(5)


# In[4]:


df.head()


# In[5]:


df.groupby('Category').describe()


# In[6]:


df['spam'] = df['Category'].apply(lambda x: 1 if x == "spam" else 0) 


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2, random_state=2)


# In[8]:


cv = CountVectorizer()
X_train_transform = cv.fit_transform(X_train.values)
X_test_transform = cv.transform(X_test)


# In[9]:


model = MultinomialNB()
model.fit(X_train_transform, y_train)
y_pred = model.predict(X_test_transform)


# In[10]:


model.score(X_test_transform, y_test)


# In[11]:


text = ['Free entry in 2 a wkly comp to win FA Cup fina']
model.predict(cv.transform(text))

