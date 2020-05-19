# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:56:58 2019

@author: wmy
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.impute import SimpleImputer
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.preprocessing import PolynomialFeatures
#import warnings
#import sys
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import roc_auc_score
import gc
from Kaggle.mymodule.dist_cate import dist_cate

num_cores = multiprocessing.cpu_count()
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/test.csv')
train = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/train.csv')
target=train['target']
ID_code=train['ID_code'].values
features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]
#------------test my module----------
#dc=dist_cate()
#dc.fit(train[features],target)
#train=dc.fit_transform(train[features],target)
#test=dc.transform(test[features])

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y
#def agg_feature(df):
#    df['sum'] = df[features].sum(axis=1)  
#    df['min'] = df[features].min(axis=1)
#    df['max'] = df[features].max(axis=1)
#    df['mean'] = df[features].mean(axis=1)
#    df['std'] = df[features].std(axis=1)
#    df['skew'] = df[features].skew(axis=1)
#    df['kurt'] = df[features].kurtosis(axis=1)
#    df['med'] = df[features].median(axis=1)
#    df['quan_1'] = np.quantile(df[features],0.01)
#    df['quan_2'] = np.quantile(df[features],0.05)
#    df['quan_3'] = np.quantile(df[features],0.95)
#    df['quan_4'] = np.quantile(df[features],0.99)
#    return df
#train=agg_feature(train)
#test=agg_feature(test)
#train=train[features].round()
#test=test[features].round()
#pf = PolynomialFeatures(degree=2, interaction_only=False,  
#                        include_bias=False)
#train = pf.fit_transform(train[features]) 
#test = pf.transform(test[features])  
#train_backup=train.copy()
#train=train[features]**3
#train.columns=['cube_'+c for c in features]
#train=pd.concat([train,train_backup[features]],axis=1)
#test_backup=test.copy()
#test=test[features]**3
#test.columns=['cube_'+c for c in features]
#test=pd.concat([test,test_backup[features]],axis=1)
dc=dist_cate()#draw=True,save_path='./Kaggle/santander-customer-transaction-prediction/split.png')
train=dc.fit_transform(train[features],target)
train=pd.concat([target,train],axis=1)
train.to_csv('./Kaggle/santander-customer-transaction-prediction/train_segment.csv',index=False)
test_cate=dc.transform(test[features])
train_cate.columns=['cate_'+c for c in features]
test_cate.columns=['cate_'+c for c in features]
uniq_val=train_cate.nunique().values
train_cate=train_cate.loc[:,uniq_val<100]
test_cate=test_cate.loc[:,uniq_val<100]
train=pd.concat([train,train_cate],axis=1)
test=pd.concat([test,test_cate],axis=1)
used_features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]
#train['var_68']=train['var_68'].round()
#train[used_features]=np.exp(train[used_features])
#params = {
#    'bagging_freq': 5,
#    'bagging_fraction': 0.33,
#    'boost_from_average':'false',
#    'boost': 'gbdt',
#    'feature_fraction': 0.04,
#    'learning_rate': 0.008,
#    'max_depth': -1,
#    'metric':'auc',
#    'seed': 2019,
#    'feature_fraction_seed': 2019,
#    'bagging_seed': 2019,
#    'data_random_seed': 2019,
#    'min_data_in_leaf': 80,
#    'min_sum_hessian_in_leaf': 10.0,
#    'num_leaves': 13,
#    'num_threads': num_cores,
#    'tree_learner': 'serial',
#    'objective': 'binary',
#    'verbosity': 1
#}
params = {#magic
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    "metric" : "auc",
    'objective': 'binary',
    'verbosity': 1}
params = {#augment
    "objective" : "binary",
    'max_bin':600,
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : 4590,
    "verbosity" : 1,
    "seed": 4590
}
params = {#test
    'bagging_freq': 5,
#    'max_bin':400,
    'bagging_fraction': 0.6,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'min_data_in_leaf': 400,
#    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 6,
    'num_threads': 6,
#    'tree_learner': 'serial',
        "lambda_l1" : 0.5,
    "lambda_l2" : 0.5,
    "metric" : "auc",
    'objective': 'binary',
    'verbosity': 1}
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=4590)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()
#a=pd.concat([train_backup.min(axis=0),train_backup.max(axis=0)],axis=1)
#LOO_list=['ProductName','SmartScreen','CityIdentifier','OsBuildLab','regionIdentifier','Platform','Processor','OsPlatformSubRelease','SkuEdition','Census_ProcessorClass','Census_PrimaryDiskTypeName','Census_ProcessorManufacturerIdentifier','Census_OSWUAutoUpdateOptionsName','Census_ActivationChannel','Census_MDC2FormFactor','Census_DeviceFamily','Census_FlightRing','Census_OSArchitecture','SmartScreen','Census_PowerPlatformRoleName','Census_OSBranch','Census_OSSkuName','Census_OSInstallTypeName','EngineVersion_3']
#LOO=LeaveOneOutEncoder(cols=used_features,randomized=True,handle_unknown='ignore')
#LOO.fit(train[used_features],target)
#test=LOO.transform(test[used_features])
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))
    train_x=train.iloc[trn_idx].reset_index(drop=True)
    valid_x=train.iloc[val_idx].reset_index(drop=True)
    target_train=target.iloc[trn_idx].reset_index(drop=True)
    target_valid=target.iloc[val_idx].reset_index(drop=True)
#    LOO=LeaveOneOutEncoder(cols=used_features,randomized=True,handle_unknown='ignore')
#    train_x=LOO.fit_transform(train_x[used_features],target_train)
#    valid_x=LOO.transform(valid_x[used_features])  
    trn_data = lgb.Dataset(train_x[used_features],
                           label=target_train,
#                           categorical_feature=cat_cols
                          )
    val_data = lgb.Dataset(valid_x[used_features],
                           label=target_valid,
#                           categorical_feature=cat_cols
                          )

    num_round = 1000000
    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=500,
                    early_stopping_rounds = 1000)
    
    oof[val_idx] = clf.predict(valid_x[used_features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = used_features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[used_features], num_iteration=clf.best_iteration) / folds.n_splits
#===========================================
#    N = 5
#    p_valid,yp = 0,0
#    for i in range(N):
#        X_t, y_t = augment(train_x[used_features].values, target_train.values)
#        X_t = pd.DataFrame(X_t)
#        X_t = X_t.add_prefix('var_')
#    
#        trn_data = lgb.Dataset(X_t, label=y_t)
#        val_data = lgb.Dataset(valid_x[used_features], label=target_valid)
##        evals_result = {}
#        lgb_clf = lgb.train(params,
#                        trn_data,
#                        100000,
#                        valid_sets = [trn_data, val_data],
#                        early_stopping_rounds=3000,
#                        verbose_eval = 5000,
##                        evals_result=evals_result
#                       )
#        oof[val_idx] += lgb_clf.predict(valid_x[used_features], num_iteration=lgb_clf.best_iteration)/N
#        predictions += lgb_clf.predict(test[used_features], num_iteration=lgb_clf.best_iteration) / (folds.n_splits*N)
print("CV score: {:<8.5f}".format(roc_auc_score(target,oof)))

f_noimp_avg = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False))
f=plt.figure(figsize=(14,6))
plt.plot(f_noimp_avg)
plt.xticks(rotation=-45)
#
used_features=f_noimp_avg.index[:200].tolist()
#used_features=[c for c in used_features if c[0]=='c']
#used_features=features+used_features[:50]
#used_features=used_features.loc[:200,'feature'].tolist()

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = predictions
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/sub_mylgbm_aug_600bin_15cv_{}.csv'.format(roc_auc_score(target,oof)), index=False)

df_train= pd.DataFrame({"ID_code":ID_code})
df_train['M_lgbm']=oof
df_train.to_csv('./Kaggle/santander-customer-transaction-prediction/sub_mylgbm_aug_600bin_15cv_train.csv', index=False)

#sns.distplot((train['var_81'].loc[target==1])**3)
#sns.distplot((train['var_81'].loc[target==0])**3)
#a=train.loc[target==1,used_features].mean()
#b=train.loc[target==0,used_features].mean()
#f,ax=plt.subplots(40,5,figsize=(40*5,5*5))
#for i in range(40):
#    for j in range(5):
#        sns.distplot(train.loc[target==1,used_features[i*2+j]],ax=ax[i,j])
#        sns.distplot(train.loc[target==0,used_features[i*2+j]],ax=ax[i,j])
#        plt.title(used_features[i*2+j])
#plt.savefig('/gpfs/home/mw15m/santander-customer-transaction-prediction/distri.png')
x_train,x_valid,y_train,y_valid=train_test_split(train,target,test_size=0.2,stratify=target,random_state=4590,shuffle=True)
trn_data = lgb.Dataset(x_train[used_features], label=y_train)
val_data = lgb.Dataset(x_valid[used_features], label=y_valid)
#        evals_result = {}
lgb_clf = lgb.train(params,
                trn_data,
                100000,
                valid_sets = [trn_data, val_data],
                early_stopping_rounds=3000,
                verbose_eval = 5000,
#                        evals_result=evals_result
               )
oof = lgb_clf.predict(x_valid[used_features], num_iteration=lgb_clf.best_iteration)
predictions = lgb_clf.predict(test[used_features], num_iteration=lgb_clf.best_iteration)
roc_auc_score(y_valid,oof)
#./Kaggle/