import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

# https://www.kaggle.com/samratp/santander-value-prediction-xgb-and-lightgbm

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
col = [c for c in train.columns if c not in ['ID', 'target']]
print(train.shape, test.shape)

scl = preprocessing.StandardScaler()

#### Check if there are any NULL values in Train Data
print("Total Train Features with NaN Values = " + str(train.columns[train.isnull().sum() != 0].size))
# if (train.columns[train.isnull().sum() != 0].size):
    # print("Features with NaN => {}".format(list(train_df.columns[train.isnull().sum() != 0])))
    # train[train.columns[train.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
#### Check if there are any NULL values in Test Data
print("Total Test Features with NaN Values = " + str(test.columns[test.isnull().sum() != 0].size))
# if (test.columns[test.isnull().sum() != 0].size):
    # print("Features with NaN => {}".format(list(test.columns[test.isnull().sum() != 0])))
    # test[test.columns[test.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)

train = train.fillna(train.mean())
test = test.fillna(test.mean())

#### Check if there are any NULL values in Train Data
print("Total Train Features with NaN Values = " + str(train.columns[train.isnull().sum() != 0].size))
# if (train.columns[train.isnull().sum() != 0].size):
    # print("Features with NaN => {}".format(list(train_df.columns[train.isnull().sum() != 0])))
    # train[train.columns[train.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
#### Check if there are any NULL values in Test Data
print("Total Test Features with NaN Values = " + str(test.columns[test.isnull().sum() != 0].size))
# if (test.columns[test.isnull().sum() != 0].size):
    # print("Features with NaN => {}".format(list(test.columns[test.isnull().sum() != 0])))
    # test[test.columns[test.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)



def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))


x1, x2, y1, y2 = model_selection.train_test_split(
    train[col], train.target.values, test_size=0.20, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=7)
model.fit(scl.fit_transform(x1), y1)
print(rmsle(y2, model.predict(scl.transform(x2))))

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(
    by=['importance'], ascending=[False])[:480]['feature'].values

test['target_lgb'] = 0.0
test['target_xgb'] = 0.0
test['target_cb'] = 0.0
test['target_rfr'] = 0.0
folds = 5
for fold in range(folds):
    print("range loop:",fold)
    
    x1, x2, y1, y2 = model_selection.train_test_split(
        train[col], np.log1p(train.target.values), test_size=0.20, random_state=fold)
        
    # LightGBM
    print("start light gbm")
    params = {'learning_rate': 0.02, 'max_depth': 13, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'rmse',
              'is_training_metric': True, 'num_leaves': 12**2, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'seed': fold}
    model = lgb.train(params, lgb.Dataset(x1, label=y1), 3000, lgb.Dataset(
        x2, label=y2), verbose_eval=200, early_stopping_rounds=100)
    test['target_lgb'] += np.expm1(model.predict(test[col],
                                                 num_iteration=model.best_iteration))
    # XGB
    print("start light xgb")
    watchlist = [(xgb.DMatrix(x1, y1), 'train'),
                 (xgb.DMatrix(x2, y2), 'valid')]

    params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.005, 'max_depth': 10,
              'subsample': 0.7, 'colsample_bytree': 0.5, 'alpha': 0, 'silent': True, 'random_state': fold}
    model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist,
                      maximize=False, verbose_eval=200, early_stopping_rounds=100)
    test['target_xgb'] += np.expm1(model.predict(xgb.DMatrix(test[col]),
                                                 ntree_limit=model.best_ntree_limit))

    #CatBoostRegressor
    print("start light CatBoostRegressor")    
    model = CatBoostRegressor(iterations=500,
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
    model.fit(x1, y1,
             eval_set=(x2, y2),
             use_best_model=True,
             verbose=True)
    test['target_cb'] += np.expm1(model.predict(test[col]))

    # RandomForestRegressor
    print("start light RandomForestRegressor")    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(x1, y1)
    test['target_rfr'] += np.expm1(model.predict(test[col]))


test['target_lgb'] /= folds
test['target_xgb'] /= folds
test['target_cb'] /= folds
test['target_rfr'] /= folds
test['target'] = (test['target_lgb'] + test['target_xgb'] + test['target_cb'] + test['target_rfr'])/4
test[['ID', 'target']].to_csv('submission1.csv', index=False)
