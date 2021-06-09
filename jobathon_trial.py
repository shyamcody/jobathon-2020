#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 01:34:13 2021

@author: shyambhu.mukherjee
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
train_data = pd.read_csv('/home/shyambhu.mukherjee/Downloads/train_jobathon.csv')
test_data = pd.read_csv('/home/shyambhu.mukherjee/Downloads/test_jobathon.csv')
print(train_data.columns)
train_data = train_data.drop(['ID'],axis = 1)
test_ids = list(test_data['ID'])
test_data = test_data.drop(['ID'],axis = 1)

print(train_data.Response.describe())
print(train_data.Response.value_counts())
print(train_data.isna().sum())
from xgboost import XGBClassifier

print(train_data.columns)
cols = ['City_Code','Region_Code','Accomodation_Type', 'Reco_Insurance_Type','Holding_Policy_Type',
        'Reco_Policy_Cat','Health Indicator']
region_count = train_data['Region_Code'].value_counts()
replace_these = list(region_count[region_count<3].index)
train_data['Region_Code'] = train_data['Region_Code'].replace(replace_these,'OTHERS')
region_count = test_data['Region_Code'].value_counts()
replace_these = list(region_count[region_count<3].index)
test_data['Region_Code'] = test_data['Region_Code'].replace(replace_these,'OTHERS')
for col in cols:
    train_data[col] = train_data[col].fillna('UA')
    test_data[col] = test_data[col].fillna('UA')
    uniques = list(set(list(train_data[col].unique())).intersection(set(list(test_data[col].unique()))))
    for unique in uniques:
        train_data['Is_'+col+'_equal_'+str(unique)] = train_data[col].apply(lambda x: (x==unique)*1.0)
        test_data['Is_'+col+'_equal_'+str(unique)] = test_data[col].apply(lambda x: (x==unique)*1.0) 
    train_data = train_data.drop(col,axis = 1)
    test_data = test_data.drop(col,axis = 1)

print(train_data.columns)
def yes_no(x):
    if x== 'Yes': return 1
    return 0
def digited(x):
    dicts = {'1.0':1, '14.0':14,'3.0':3,'7.0':7, 
             '0.0':0, '2.0':2, '11.0':11,'6.0':6, 
             '4.0':4,'8.0':8, '9.0':9, '10.0':10, 
             '5.0':5, '12.0':12, '13.0':13}
    return dicts[x]

def divider(x,y):
    if y==0: return 0
    return x/y

def extreme_z(x):
    if x< -1.5: 
        return 1
    elif x>4:
        return 1
    else:
        return 0


def feature_addition(data):
    data['Holding_Policy_Duration'] = data['Holding_Policy_Duration'].fillna('0.0')
    data['Holding_Policy_Duration_more_than_14'] = data['Holding_Policy_Duration'].apply(lambda x: x=='14+')
    data['Holding_Policy_Duration'] = data['Holding_Policy_Duration'].replace('14+','14.0')
    data['Holding_Policy_Duration'] = data['Holding_Policy_Duration'].apply(lambda x: digited(x))
    data['Is_Spouse'] = data['Is_Spouse'].apply(lambda x: yes_no(x))
    data['Age_difference'] = data['Upper_Age'] - data['Lower_Age']
    data['Is_Age_difference_0'] = data['Age_difference'].apply(lambda x: x==0)
    data['policy_duration_difference_ratio'] = data.apply(lambda x: divider(x['Age_difference'],x['Holding_Policy_Duration']),axis = 1)
    data['Reco_Policy_Premium_z'] = data['Reco_Policy_Premium'].apply(lambda x: (x/6500 - 2))
    data['Reco_Policy_extreme'] = data['Reco_Policy_Premium_z'].apply(extreme_z)
    return data


train_data = feature_addition(train_data)
test_data = feature_addition(test_data)

pca = PCA(n_components = 400)
#train_data['Holding_policy_Duration'] = train_data['Holding_Policy_Duration'].astype('float64')
#test_data['Holding_policy_Duration'] = test_data['Holding_Policy_Duration'].astype('float64')
X = train_data.drop('Response',axis = 1)
Y = train_data['Response']

## making pca classifier to improve the result by compactifying the features.
pca = PCA(n_components = 300)
X = pca.fit_transform(X)
test_data = pca.transform(test_data)
print(sum(pca.explained_variance_ratio_))
from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test = tts(X,Y,test_size = 0.2,random_state = 42,
                                    stratify = Y)
classifier = XGBClassifier(
                           #n_estimators = 2000,
                           #learning_rate = 0.01,
                           scale_pos_weight = 4)
#X_train['Holding_policy_Duration'] = pd.to_numeric(X_train['Holding_Policy_Duration'])
#X_test['Holding_policy_Duration'] = pd.to_numeric(X_test['Holding_Policy_Duration'])
"""
0.72
0.6115
"""
classifier.fit(X_train,Y_train)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
Y_pred_train = classifier.predict(X_train)
Y_pred_test = classifier.predict(X_test)
print(classification_report(Y_train,Y_pred_train))
print(classification_report(Y_test,Y_pred_test))
print(roc_auc_score(Y_train,Y_pred_train))
print(roc_auc_score(Y_test,Y_pred_test))

#from catboost import CatBoostClassifier
#cbc = CatBoostClassifier()
#cbc.fit(X_train,Y_train)
#Y_pred_train = cbc.predict(X_train)
#Y_pred_test = cbc.predict(X_test)
#print(classification_report(Y_train,Y_pred_train))
#print(classification_report(Y_test,Y_pred_test))
#print(roc_auc_score(Y_train,Y_pred_train))
#print(roc_auc_score(Y_test,Y_pred_test))

#from sklearn.ensemble import AdaBoostClassifier as ADBC
#abc = ADBC(n_estimators = 1000,learning_rate = 0.01,random_state = 24)
#abc.fit(X_train,Y_train)
#Y_pred_train = abc.predict(X_train)
#Y_pred_test = abc.predict(X_test)
#print(classification_report(Y_train,Y_pred_train))
#print(classification_report(Y_test,Y_pred_test))
#print(roc_auc_score(Y_train,Y_pred_train))
#print(roc_auc_score(Y_test,Y_pred_test))

from sklearn.ensemble import RandomForestClassifier as RFC
rfc = RFC(n_estimators = 128,max_depth = 14,min_samples_split = 5,
          class_weight = 'balanced',n_jobs = -1)
rfc.fit(X_train,Y_train)
Y_pred_train = rfc.predict(X_train)
Y_pred_test = rfc.predict(X_test)
print(classification_report(Y_train,Y_pred_train))
print(classification_report(Y_test,Y_pred_test))
print("rfc classifier performance:")
print(roc_auc_score(Y_train,Y_pred_train))
print(roc_auc_score(Y_test,Y_pred_test))
"""
0.6602465939154291
0.6007656360597537
"""
#df = pd.DataFrame()
#df['columns'] = list(X_train.columns)
#df['importances'] = rfc.feature_importances_
#sum(pca.explained_variance_ratio_)
pred_df = pd.DataFrame()
pred_df['ID'] = test_ids
dataframe = pd.DataFrame(classifier.predict_proba(test_data),columns = ['0','1'])
pred_df['Response'] = dataframe['1']
pred_df.to_csv('jobathon_basic_prediction.csv',index = False)

from sklearn.ensemble import ExtraTreesClassifier as ETC
etc = ETC(n_estimators = 128,max_depth = 16,min_samples_split = 30,
          class_weight = 'balanced',n_jobs = -1)
etc.fit(X_train,Y_train)
Y_pred_train = etc.predict(X_train)
Y_pred_test = etc.predict(X_test)
print(classification_report(Y_train,Y_pred_train))
print(classification_report(Y_test,Y_pred_test))
print("extratrees classifier performance")
print(roc_auc_score(Y_train,Y_pred_train))
print(roc_auc_score(Y_test,Y_pred_test))


pred_df = pd.DataFrame()
pred_df['ID'] = test_ids
dataframe = pd.DataFrame(etc.predict_proba(test_data),columns = ['0','1'])
pred_df['Response'] = dataframe['1']
pred_df.to_csv('jobathon_etclassifier_new_pca_prediction.csv',index = False)

from lightgbm import LGBMClassifier as LGBC
lgbc = LGBC(boosting_type = 'dart',n_estimators = 1500,
            learning_rate = 0.02,colsample_bytree = 0.8,
            subsample = 0.6,min_child_samples = 20,
            class_weight = 'balanced',n_jobs = -1)
lgbc.fit(X_train,Y_train)
Y_pred_train = lgbc.predict(X_train)
Y_pred_test = lgbc.predict(X_test)
print(classification_report(Y_train,Y_pred_train))
print(classification_report(Y_test,Y_pred_test))
print("lgbc classifier performance:")
print(roc_auc_score(Y_train,Y_pred_train))
print(roc_auc_score(Y_test,Y_pred_test))


pred_df = pd.DataFrame()
pred_df['ID'] = test_ids
dataframe = pd.DataFrame(lgbc.predict_proba(test_data),columns = ['0','1'])
pred_df['Response'] = dataframe['1']
pred_df.to_csv('jobathon_lgbclassifier_final_prediction.csv',index = False)
"""
lgbc classifier performance:
0.7921842406644868
0.6633504280563104
"""
"""
from sklearn.ensemble import StackingClassifier as ST
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import GradientBoostingClassifier as GBC
stack = ST([('rf',RFC(n_estimators = 128,max_depth = 14,min_samples_split = 5,
          class_weight = 'balanced',n_jobs = -1)),
            ('etc',ETC(n_estimators = 128,max_depth = 16,min_samples_split = 5,
          class_weight = 'balanced',n_jobs = -1)),
            #('gbc',GBC(learning_rate=0.05,n_estimators=1000, 
             #          subsample=0.5,max_depth=8,n_jobs = -1)),
            ('knn',KNC())],
          final_estimator= (ETC(n_estimators = 128,max_depth = 16,min_samples_split = 30,
          class_weight = 'balanced',n_jobs = -1))
           )

#0.5260807121398227
#0.5166083254318549
#for logistic final model
#0.8241498222171529
#0.6592372386490034
#for etc as the final estimator
#historically we have seen that stacking classifier performs slightly worse
#than the lgbc classifier. We have the lgbc > stacking>etc>xgboost>randomforest

stack.fit(X_train,Y_train)
Y_pred_train = stack.predict(X_train)
Y_pred_test = stack.predict(X_test)
print(classification_report(Y_train,Y_pred_train))
print(classification_report(Y_test,Y_pred_test))
print(roc_auc_score(Y_train,Y_pred_train))
print(roc_auc_score(Y_test,Y_pred_test))

pred_df = pd.DataFrame()
pred_df['ID'] = test_ids
dataframe = pd.DataFrame(stack.predict_proba(test_data),columns = ['0','1'])
pred_df['Response'] = dataframe['1']
pred_df.to_csv('jobathon_stackclassifier_pca_prediction.csv',index = False)
#stacking fit
"""