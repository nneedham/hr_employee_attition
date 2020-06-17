# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:42:36 2020

@author: needh
"""


import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#Choose the relevant columns
#Since there were many correlations in the numerical variables, we are going to
# look at the categorical
df_model = df[['Attrition',
              'BusinessTravel',
              'Department',
              'Education',
              'EducationField',
              'EnvironmentSatisfaction',
              'JobInvolvement',
              'JobLevel',
              'JobRole',
              'JobSatisfaction',
               'MaritalStatus',
              'OverTime',
              'PerformanceRating',
               'RelationshipSatisfaction',
              ]]

#uncomment to look at all variables
#df_model = df

df_dum = pd.get_dummies(df_model)

#Train the test splits
from sklearn.model_selection import train_test_split
X =df_dum.drop(['Attrition_Yes', 'Attrition_No'], axis=1) 
y = df_dum.Attrition_Yes.values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic Regrssion
# We want to do a logistic regression because of the binary result of attrition
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
lrscore = lr.score(X_test, y_test)
print('Logistic Regression accuracy: ', lrscore)

#Discriminant Analysis
#This is used when input variables do not result in proportional changes to 
# the output analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
pred_lda = lda.predict(X_test)
lda_score = lda.score(X_test, y_test)
print('Linear Discriminant accuracy: ', lda_score)

#Support Vector Machine SVM
#This will show the difference between classes... so we can see the difference
# between retention and attriting
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
s_vm = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
s_vm.fit(X_train, y_train)
pred_s_vm = s_vm.predict(X_test)
s_vm_score = s_vm.score(X_test, y_test)
print('Support Vector Machine accuracy: ', s_vm_score)

#Random Forest
# can predict likelihood between discrete problems
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=10, random_state=0)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
rfc_score = rfc.score(X_test, y_test)
print('Random Forect accuracy: ', rfc_score)

#Since SVM is highest let's dig deeper into that
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_s_vm))

#Looking first at employees retained (0) we see that precision is .89... that is 
# to say that when it predicts an employee will stay it is correct 89% of the time
# the recall is 98%... that is to say that it correctly identifies 98% of all employees retained

#Looking next at employees attritted we see that the precision is .64... that is 
# to say that when it predicts an employee will attrit it is correct 64% of the time
# the recall is .23... that is to say that it correctly identifies only 23% of 
#all employees who attrit

