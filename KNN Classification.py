#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors with Python
# ## you have been given a classified data set from a company. They have hidden the feature column namew but have given you the data and target classes. we'll try to use to create a model that directly predicts a class for a new data point based off of the feature

# In[90]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
import seaborn as sns


# In[91]:


df=pd.read_csv("Classified Data.csv",index_col=0)


# In[92]:


df.head()


# # standardize the variables

# In[93]:


from sklearn.preprocessing import StandardScaler


# In[94]:


scaler=StandardScaler()


# In[95]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[96]:


scaled_features= scaler.transform(df.drop('TARGET CLASS', axis=1))


# In[ ]:





# In[97]:


df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[98]:


## Train Test Split


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


X_train,X_test,y_train,y_test=train_test_split(scaled_features,df['TARGET CLASS'])


# # using KNN

# In[101]:


from sklearn.neighbors import KNeighborsClassifier


# In[102]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[103]:


knn.fit(X_train,y_train)


# In[104]:


pred=knn.predict(X_test)


# # prediction and Evaluation

# In[105]:


from sklearn.metrics import classification_report,confusion_matrix


# In[106]:


print(confusion_matrix(y_test,pred))


# In[107]:


print(classification_report(y_test,pred))


# # Choosing a k Value

# In[109]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    


# In[111]:


plt.figure(figsize=(6,10))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[115]:


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print("with=1")
print(confusion_matrix(y_test,pred))
print("\n")
print(classification_report(y_test,pred))


# In[117]:


knn=KNeighborsClassifier(n_neighbors=18)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[ ]:




