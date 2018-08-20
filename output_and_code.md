

```python
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
```


```python
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target
```

# splitting into training and test data


```python
X_train, X_test,y_train,y_test = train_test_split(features,targets,test_size=0.25,random_state=33)
```

# scaling the features


```python
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
```

    10.202898004591216 -4.6670204084548 2.4703870638462586e-15 2.9177492036731256 -1.931470986413033 3.5855223803197665e-16
    


```python
def train_and_evaluate(clf, X_train, y_train):
    
    clf.fit(X_train, y_train)
    
    print ("Coefficient of determination on training set:",clf.score(X_train, y_train))
    
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
```


```python
extraTreeRegressor = ensemble.ExtraTreesRegressor(n_estimators=10,random_state=42)
train_and_evaluate(extraTreeRegressor,X_train=X_train,y_train=y_train)
```

    Coefficient of determination on training set: 1.0
    Average coefficient of determination using 5-fold crossvalidation: 0.8617589783439273
    


```python
important=zip(extraTreeRegressor.feature_importances_,boston.feature_names)
print (sorted(important))
```

    [(0.005043853202755884, 'ZN'), (0.015142513715149682, 'B'), (0.017052578400506287, 'AGE'), (0.018941821085751577, 'RAD'), (0.023602561777571307, 'CHAS'), (0.025733049004581798, 'CRIM'), (0.03187416223510046, 'NOX'), (0.03440564493930893, 'INDUS'), (0.039713133345196064, 'DIS'), (0.046618521397262996, 'TAX'), (0.09951180149276224, 'PTRATIO'), (0.28421522796368465, 'LSTAT'), (0.3581451314403682, 'RM')]
    

#### RM, LSTAT, PTRATIO are the most significant features


```python
y_pred=extraTreeRegressor.predict(X_test)   
# print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test,y_pred)),"\n")
# print ("Classification report")
# print (metrics.classification_report(y_test,y_pred),"\n")
# print ("Confusion matrix")
# print (metrics.confusion_matrix(y_test,y_pred),"\n")
print ("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y_test,y_pred)),"\n")
```

    Coefficient of determination:0.802 
    
    

# Random Forest Regressor


```python
randomForestRegressor = ensemble.RandomForestRegressor(n_estimators=10,criterion='mse',max_features=3,random_state=42,verbose=1,max_depth=5)
randomForestRegressor.fit(X_train,y_train)
```

    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features=3, max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=10, n_jobs=1, oob_score=False, random_state=42,
               verbose=1, warm_start=False)




```python
print(randomForestRegressor.feature_importances_)
```

    [0.03144617 0.05028242 0.05817739 0.0095859  0.09215994 0.24766723
     0.07692349 0.04740923 0.0017567  0.02596756 0.04030291 0.02150828
     0.29681277]
    


```python
train_and_evaluate(randomForestRegressor,X_train=X_train,y_train=y_train)
```

    Coefficient of determination on training set: 0.9039793988049013
    Average coefficient of determination using 5-fold crossvalidation: 0.8084860189448481
    

    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    


```python
y_pred=randomForestRegressor.predict(X_test)   
# print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test,y_pred)),"\n")
# print ("Classification report")
# print (metrics.classification_report(y_test,y_pred),"\n")
# print ("Confusion matrix")
# print (metrics.confusion_matrix(y_test,y_pred),"\n")
print ("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y_test,y_pred)),"\n")
```

    Coefficient of determination:0.738 
    
    

    [Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.0s finished
    
