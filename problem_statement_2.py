
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cross_validation import *
from sklearn import ensemble
from sklearn import metrics


# In[2]:


boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target


# # splitting into training and test data

# In[9]:


X_train, X_test,y_train,y_test = train_test_split(features,targets,test_size=0.25,random_state=33)


# # scaling the features

# In[14]:


scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(np.reshape(y_train,(-1,1)))

X_train = scalerX.transform(X_train)
y_train = scalery.transform(np.reshape(y_train,(-1,1)))
X_test = scalerX.transform(X_test)
y_test = scalery.transform(np.reshape(y_test,(-1,1)))

print (np.max(X_train), np.min(X_train), np.mean(X_train), np.max(y_train), np.min(y_train), np.mean(y_train))
# converting reshaped Y vector into array again
y_train=y_train.flatten(order='C')
y_test=y_test.flatten(order='C')


# In[16]:


def train_and_evaluate(clf, X_train, y_train):
    
    clf.fit(X_train, y_train)
    
    print ("Coefficient of determination on training set:",clf.score(X_train, y_train))
    
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))


# In[18]:


extraTreeRegressor = ensemble.ExtraTreesRegressor(n_estimators=10,random_state=42)
train_and_evaluate(extraTreeRegressor,X_train=X_train,y_train=y_train)


# In[20]:


important=zip(extraTreeRegressor.feature_importances_,boston.feature_names)
print (sorted(important))


# #### RM, LSTAT, PTRATIO are the most significant features

# In[24]:


y_pred=extraTreeRegressor.predict(X_test)   
# print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test,y_pred)),"\n")
# print ("Classification report")
# print (metrics.classification_report(y_test,y_pred),"\n")
# print ("Confusion matrix")
# print (metrics.confusion_matrix(y_test,y_pred),"\n")
print ("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y_test,y_pred)),"\n")


# # Random Forest Regressor

# In[26]:


randomForestRegressor = ensemble.RandomForestRegressor(n_estimators=10,criterion='mse',max_features=3,random_state=42,verbose=1,max_depth=5)
randomForestRegressor.fit(X_train,y_train)


# In[27]:


print(randomForestRegressor.feature_importances_)


# In[28]:


train_and_evaluate(randomForestRegressor,X_train=X_train,y_train=y_train)


# In[30]:


y_pred=randomForestRegressor.predict(X_test)   
# print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test,y_pred)),"\n")
# print ("Classification report")
# print (metrics.classification_report(y_test,y_pred),"\n")
# print ("Confusion matrix")
# print (metrics.confusion_matrix(y_test,y_pred),"\n")
print ("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y_test,y_pred)),"\n")

