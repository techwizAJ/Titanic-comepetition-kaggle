"""
# -*- coding: utf-8 -*-

@author: techwiz
Created on Sun May 27 14:47:20 2018
"""
import pandas as pd

train_set = pd.read_csv("train.csv")
test_set =  pd.read_csv("test.csv")

""" Exploratory Data Analysis """
train_set['Sex'].value_counts()
train_set['Age'].value_counts()
train_set['Embarked'].value_counts()
train_set.isnull().values.any()
train_set.isnull().sum().sum()
train_set.describe()

# Selecting required features from training dataset
train_set.drop('PassengerId', axis=1, inplace= True)
train_set.drop('Name' , axis=1,inplace=True)
train_set.drop('Cabin' , axis =1 , inplace=True)
train_set.drop('Ticket',axis=1, inplace = True)
test_set.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
#Encoding Categorial Data
train_set['Age'].hist(bins=30)
train_set['Fare'].hist(bins=30)

# impute missing values
"""
Losing Data Distribution by imputing through mean and median
train_set.fillna(train_set.mean(),inplace=True)
train_set.isnull().values.any()
test_set.fillna(train_set.mean(),inplace=True)
test_set.isnull().values.any()
"""
# imputing data with outliners
train_set['Age'].fillna(-1,inplace=True)
train_set['Fare'].fillna(-1,inplace=True)
train_set['Embarked'].fillna('Q',inplace=True)
test_set['Age'].fillna(-1,inplace=True)
test_set['Fare'].fillna(-1,inplace=True)
test_set['Embarked'].fillna('Q',inplace=True)
#LabelEncoder
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train_set['Sex'] = lb.fit_transform(train_set['Sex'])
test_set['Sex'] = lb.fit_transform(test_set['Sex'])
lb_t = LabelEncoder()
train_set['Embarked'] = lb_t.fit_transform(train_set['Embarked'])
test_set['Embarked'] = lb_t.fit_transform(test_set['Embarked'])

"""
train_set = pd.get_dummies(data= train_set , dummy_na = True,columns =['Sex' , 'Embarked'])
test_set = pd.get_dummies(data= test_set , dummy_na = True,columns =['Sex' , 'Embarked'])
train_set.drop('Sex_nan',axis=1,inplace=True)
test_set.drop('Sex_nan',axis=1,inplace=True)
"""


# Selecting Features and target
X = train_set.iloc[:,1:13].values
y = train_set.iloc[:,0].values
X_test = test_set.iloc[:,:].values

"""
#Validating Model for Parameter tuning 
from sklearn.model_selection import train_test_split
X_train , X_validate , y_train , y_validate = train_test_split(X,y,test_size=0.18,random_state=42)

#Now Appling Various ML Models For Classification 

#Feature Scaling , testing differnt scalers and their effect on data distibution
#Using  Min Max Scalar
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0.5,0.95))
train_set = scaler.fit_transform(train_set)
test_set = scaler.fit_transform(test_set)
train_set['Age'].hist(bins=30)

#testing differnt scalers 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_set = sc_X.fit_transform(train_set)
test_set = sc_X.fit_transform(test_set)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=4,random_state=42,warm_start=True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_validate)

import xgboost as xg
classifier = xg.XGBClassifier()
classifier.fit(X_train,y_train)
y_predict_xg = classifier.predict(X_validate)

#metrics
from sklearn.metrics import confusion_matrix
cnf = confusion_matrix(y_validate,y_pred)
cnf1 = confusion_matrix(y_validate,y_predict_xg)
"""

#Feature Scaling , testing differnt scalers and their effect on data distibution

#Using  Min Max Scalar
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0.5,0.95))
X = scaler.fit_transform(X)
X_test= scaler.transform(X_test)
train_set['Age'].hist(bins=30)

"""
#testing differnt scalers 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
"""

#using various ml models 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=4,random_state=42,warm_start=True)
clf.fit(X,y)

"""
import xgboost as xg
classifier = xg.XGBClassifier()
classifier.fit(X,y)
y_pred_xg = classifier.predict(X_test)
"""
y_predict = clf.predict(X_test)

sub = pd.read_csv('gender_submission.csv')
print(sub['Survived'].value_counts())
#submission
sub['Survived']=y_predict
sub.to_csv('submissions1.csv',index=False)
final = pd.read_csv('submissions1.csv')
print(final['Survived'].value_counts())