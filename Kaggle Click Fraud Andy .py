#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:34:30 2018

@author: andy

variable detail:
    
ip: ip address of click
app: app id for marketing
device: device type id of user mobile phone
os: os version id of user mobile phone
channel: channel id of mobile ad publisher
click_time: timestamp of click (UTC)
attributed_time: if user download the app for after clicking an ad, this is the time of the app download
is_attributed: whether one downloaded the app

"""

import pandas as pd
import matplotlib.pyplot as plt
import math as math
from sklearn.model_selection import train_test_split
import seaborn as sns


#Step 1: Read data
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']


df = pd.read_csv('train_sample.csv', dtype = dtypes, usecols = train_cols)
test = pd.read_csv('test.csv', dtype = dtypes, usecols = test_cols)

df['click_time'] = pd.to_datetime(df['click_time'])
df['click_hour'] = df['click_time'].dt.hour

test['click_time'] = pd.to_datetime(test['click_time'])
test['click_hour'] = test['click_time'].dt.hour


df_x = df.loc[:, ('app', 'device', 'os', 'channel', 'click_hour')]
df_y = df.loc[:, ('is_attributed')]

test_x = test.loc[:,  ('app', 'device', 'os', 'channel', 'click_hour')]

#Step 2: Stats summary
df.describe()
unique = {i:len(df.loc[:,i].unique()) for i in train_cols} #no. of unique value in each category


#Step 3: EDA with graphs

# Unique value bar chart
uniques = [len(df[i].unique()) for i in train_cols]  #no. of unique value in each category
plt.figure(figsize=(15, 8))
sns.set(font_scale=1.2)
ax = sns.barplot(train_cols, uniques, log= True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title = 'Number of Unique values per feature')

for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha = "center")




#Step 4: Feature Engineering / Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=999)


#Step 5: Model specification

## Catboost
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

model = CatBoostClassifier(eval_metric = "AUC", learning_rate = 0.05, iterations = 1500)
model.fit(x_train, y_train, eval_set = (x_test, y_test))
y_pred = model.predict(x_test)
#model.score(x_test, np.reshape(y_test, (30000, -1)))
#model.predict_proba(x_test)



y_pred_cat = model.predict_proba(x_test)[:, 1]
fpr_rf_cat, tpr_rf_cat, _ = roc_curve(y_test, y_pred_cat)
auc(fpr_rf_cat, tpr_rf_cat)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf_cat, tpr_rf_cat, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

feature_importance = model.get_feature_importance(np.reshape(x_test, (30000,-1)), y = y_test)


## Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(x_train, y_train)
y_pred_logistic = logistic.predict(x_test)
logistic.score(x_test , y_test.values.reshape(-1,1))
logistic.predict_proba(x_test)

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logistic)
auc(fpr, tpr)

y_pred_rf = logistic.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
auc(fpr_rf, tpr_rf)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

logistic.coef_  #Feature importance

## SVM
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 0)
svm.fit(x_train, y_train)


#Step 6: Cross validation
from sklearn.model_selection import cross_val_score
accuracies_cat = cross_val_score(estimator = model, X = df_x, y = df_y, cv = 3, scoring = 'roc_auc')
accuracies_cat.mean()
accuracies_cat.std()


accuracies_logit = cross_val_score(estimator = logistic, X = df_x, y = df_y, cv = 3, scoring = 'roc_auc')
accuracies_logit.mean()


#Step 7: model optimization techniques
## Optimize using Catboost  
## DO NOT RUN! 

from sklearn.model_selection import GridSearchCV
params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[500,1000,1500],
          'learning_rate':[0.03,0.001,0.01,0.1], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200]}

model_gridsearch = CatBoostClassifier()
grid_search =  GridSearchCV(estimator = model_gridsearch,
                            param_grid = params,
                            scoring = 'roc_auc',
                            cv = 3,
                            n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train, eval_set = (x_test, y_test))
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#Kaggle Competition

model = CatBoostClassifier(l2_leaf_reg = 300, eval_metric = "AUC", learning_rate = 0.07, iterations = 1000)
model.fit(x_train, y_train, eval_set = (x_test, y_test))
#y_pred = model.predict(x_test)

border_count = 32, 
max_ctr_complexity = 3, model_size_reg = 30
leaf_estimation_iterations = 8
scale_pos_weight = 1

####Kaggle purposes####
y_pred_test = model.predict_proba(test_x)[:, 1]
output = pd.concat([test.iloc[:,0], pd.DataFrame(y_pred_test)], axis = 1)
output = output.rename(columns={0:"is_attributed"})
output.to_csv("Kaggle Ad Click Final2.csv", index = False)





