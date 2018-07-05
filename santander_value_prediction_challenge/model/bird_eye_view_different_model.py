# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm
import gc

import warnings
warnings.filterwarnings("ignore")

df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')

print('Shape of training dataset: ',df_train.shape)
df_train.head()

print('Shape of test dataset: ',df_test.shape)
df_test.head()

df_train.info()
df_test.info()


def check_missing_data(df):
    total=df.isnull().sum().sort_values(ascending=False)
    percent=((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending=False)
    return pd.concat([total,percent],axis=1,keys=['Total','Percent'])

check_missing_data(df_train).head()
check_missing_data(df_test).head()


# Checking Unique Value in each column
df_tmp=pd.DataFrame(df_train.nunique().sort_values(),columns=['num_unique_values']).reset_index().rename(columns={'index':'Column_name'})
df_tmp.head()


def col_name_with_n_unique_value(df,n):
    df1=pd.DataFrame(df.nunique().sort_values(),columns=['num_unique_values']).reset_index()
    col_name=list(df1[df1.num_unique_values==1]['index'])
    print('count of columns with only',n,'unique values are: ',len(col_name))
    return col_name

col_to_drop=col_name_with_n_unique_value(df_train,1)

df_train.drop(columns=col_to_drop,inplace=True)
df_test.drop(columns=col_to_drop,inplace=True)
print('Shape of train dataset after droping columns: ',df_train.shape)
print('Shape of test dataset after droping columns: ',df_test.shape)

# 需要fillna基于各自列
df_train.fillna(df_train.mean())
df_test.fillna(df_test.mean())

# Getting Dataset in numpy ndarray format¶
train=df_train.iloc[:,2:].values
test=df_test.iloc[:,1:].values
target=df_train.target.values
print('Shape of train: ',train.shape)
print('Shape of target: ',target.shape)
print('Shape of test: ',test.shape)

del df_train,df_test,df_tmp
gc.collect()

# Visualization of target Column


# Feature Scaling
# 会出现`Input contains NaN, infinity or a value too large for dtype('float64').`, 暂时注释掉
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0,1))
# train_sc = sc.fit_transform(train)
# test_sc = sc.transform(test)

# 避免以下改动太多，这里保持用train_sc/test_sc
train_sc = train
test_sc = test

# D. Splitting dataset into Train, val and Test set
t=np.log1p(target)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_sc, t, test_size=0.2, random_state=0)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
# print(test_sc.shape, test_sc.shape)

del train_sc,train,test
gc.collect()


Model_Summary=pd.DataFrame()

import lightgbm
train_data=lightgbm.Dataset(X_train,y_train)
valid_data=lightgbm.Dataset(X_val,y_val)
params={'learning_rate':0.01,
        'boosting_type':'gbdt',
        'objective':'regression',
        'metric':'rmse',
        'sub_feature':0.5,
        'num_leaves':180,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'min_data':50,
        'max_depth':-1,
        'reg_alpha': 0.3, 
        'reg_lambda': 0.1, 
        'min_child_weight': 10, 
        'verbose': 1,
        'nthread':5,
        'max_bin':512,
        'subsample_for_bin':200,
        'min_split_gain':0.0001,
        'min_child_samples':5
       }
lgbm = lightgbm.train(params,
                 train_data,
                 25000,
                 valid_sets=valid_data,
                 early_stopping_rounds= 80,
                 verbose_eval= False
                 )
# TODO: fix warning `[LightGBM] [Warning] No further splits with positive gain, best gain: -inf`


model_name='lightgbm'
RMSLE=np.sqrt(mean_squared_error(y_val,lgbm.predict(X_val)))
Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
Model_Summary


pred_lgbm=np.expm1(lgbm.predict(test_sc))
pred_lgbm
print('Shape of pred_lgbm: ',pred_lgbm.shape)

#CatBoostRegressor
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.1,
                             depth=5,
                             l2_leaf_reg=20,
                             bootstrap_type='Bernoulli',
                             subsample=0.6,
                             eval_metric='RMSE',
                             random_seed = 42,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=45)

cb_model.fit(X_train, y_train,
             eval_set=(X_val, y_val),
             use_best_model=True,
             verbose=True)

model_name='CatBoostRegressor'
RMSLE=np.sqrt(mean_squared_error(y_val,cb_model.predict(X_val)))
Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
Model_Summary

pred_cb=np.expm1(cb_model.predict(test_sc))
pred_cb
print('Shape of pred_cb: ',pred_cb.shape)

# RandomForestRegressor
# from sklearn.ensemble import RandomForestRegressor
# rf_model = RandomForestRegressor(n_estimators=100)
# rf_model.fit(X_train, y_train)

# model_name='RandomForestRegressor'
# RMSLE=np.sqrt(mean_squared_error(y_val,rf_model.predict(X_val)))
# Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
# Model_Summary

# pred_rf=np.expm1(rf_model.predict(test_sc))
# pred_rf

#XGBoost
from xgboost import XGBRegressor
xgb_model=XGBRegressor(max_depth=9)
xgb_model.fit(X_train, y_train)
model_name='xgboost'
RMSLE=np.sqrt(mean_squared_error(y_val,xgb_model.predict(X_val)))
Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
Model_Summary

pred_xgb=np.expm1(xgb_model.predict(test_sc))
pred_xgb
print('Shape of pred_xgb: ',pred_xgb.shape)


#Generating Submision File
Model_Summary
sub=pd.read_csv('../input/sample_submission.csv')
# sub.target=(pred_lgbm+pred_cb+pred_rf+pred_xgb)/4.0
sub.target=(pred_lgbm+pred_cb+pred_xgb)/3.0

sub.head()
sub.to_csv('bird_eye_view_sub.csv',index=False)