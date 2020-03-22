# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:52:01 2020

@author: suyog
"""

## Multiple Linear Regression for Prediction of Profit(Dependent Variable) with R&D, Admin, Marketing,State (Independent Variable)
##using R2 score method

#Import Libraries
import pandas as pd

#Play with dataset
data = pd.read_csv('50_Startups.csv')
X= data.iloc[:,:-1] # or X = data.iloc[:,0:4]
y = data.iloc[:,4] # or y = data.iloc[:,-1]

#Looking at State column, it belong to categorical values. So lets convert same to numeric
States = pd.get_dummies(X['State'],drop_first=True)
X = X.drop('State',axis=1)
X = pd.concat([X,States],axis=1)

#Divide test set and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting for the test set results
y_pred = regressor.predict(X_test)

#Score by R2
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)

















