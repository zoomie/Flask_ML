#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 08:22:24 2018

@author: andrew
"""
import pandas as pd

df = pd.read_csv('train.csv')

df = df.drop(columns = ['Cabin','Ticket','PassengerId','Name'])
df = df.dropna(subset = ['Embarked'])

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 2:5])
X[:, 2:5] = imputer.transform(X[:, 2:5])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# lets quick try using a tree!
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


clf.predict(X_test)

#################################################################################

df_test = pd.read_csv('test.csv')

pipeline(df_test)

def pipeline(df_test):
    df_test = df_test.drop(columns = ['Cabin','Ticket','PassengerId','Name'])
    df_test = df_test.dropna(subset = ['Embarked'])
    X_pipe = df.iloc[:, 1:].values
    
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer.fit(X_pipe[:, 2:5])
    X_pipe[:, 2:5] = imputer.transform(X_pipe[:, 2:5])
    
    #Label encoding
    X_pipe[:,1] = labelencoder_X_1.transform(X_pipe[:,1])
    X_pipe[:,6] = labelencoder_X_2.transform(X_pipe[:,6])
    
    X_pipe = onehotencoder.transform(X_pipe).toarray()
    X_pipe = X_pipe[:,1:]
    
    X_pipe = sc.transform(X_pipe)
    
    answers = clf.predict(X_pipe)
    
    return answers

import pickle 
pickle.dump(pipeline, open('predict.p', 'wb'))

attempt_1 = pickle.load(open('predict.p','rb'))

attempt_1(df_test)



