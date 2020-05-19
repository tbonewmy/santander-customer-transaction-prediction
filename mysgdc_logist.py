# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:41:18 2019

@author: wmy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:56:58 2019

@author: wmy
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.impute import SimpleImputer
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.metrics import roc_auc_score
import gc
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm, rankdata
#import warnings
#import sys
#import matplotlib.pyplot as plt
#import seaborn as sns


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


test = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/test.csv')
train = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/train.csv')
target=train['target']
ID_code=train['ID_code'].values
features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]
#train=train[features]
#agg_table=pd.DataFrame([train.min().values,train.max().values,train.mean().values,train.median().values,train.std().values,train.mode().values,train.nunique().values])
#agg_table=agg_table.T
#agg_table.columns=['min','max','mean','median','std','mode','nunique']
#a=train['var_68'].sort_values()
#plt.plot(a,target,'.')
#sns.distplot(b[target==1])
#sns.distplot(b[target==0])
#plt.legend(['1','0'])
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
def createfeature(dataset1,dataset2):
    for col in dataset1.columns:
        # Normalize the data, so that it can be used in norm.cdf(), 
        # as though it is a standard normal variable
#        dataset[col] = ((dataset[col] - dataset[col].mean()) 
#        / dataset[col].std()).astype('float32')
        #first order interaction
#        for c in dataset1.columns[i+1:]:
#            dataset1[col+'*'+c] = dataset1[col].values*dataset1[c].values
#            dataset2[col+'*'+c] = dataset2[col].values*dataset2[c].values
        # Square
        dataset1[col+'^2'] = dataset1[col].values **2
        dataset2[col+'^2'] = dataset2[col].values **2
        # Cube
        dataset1[col+'^3'] = dataset1[col].values **3
        dataset2[col+'^3'] = dataset2[col].values **3
        # 4th power
        dataset1[col+'^4'] = dataset1[col].values **4
        dataset2[col+'^4'] = dataset2[col].values **4
        # Cumulative percentile (not normalized)
        temp=rankdata(pd.concat([dataset1[col],dataset2[col]],axis=0)).astype('float32')
        dataset1[col+'_cp'] =temp[:len(dataset1)]
        dataset2[col+'_cp'] =temp[len(dataset1):]
        del temp
        gc.collect()
#        dataset[col+'_cp'] = rankdata(dataset[col]).astype('float32')
    
        # Cumulative normal percentile
        dataset1[col+'_cnp'] = norm.cdf(dataset1[col],dataset1[col].mean(),dataset1[col].std()).astype('float32')
        dataset2[col+'_cnp'] = norm.cdf(dataset2[col],dataset1[col].mean(),dataset1[col].std()).astype('float32')
    return dataset1,dataset2
train,test=createfeature(train[features],test[features])
#from scipy.stats import ttest_ind
#from scipy.stats import ttest_rel
#tscore=np.zeros(train.shape[1])
#for i,c in enumerate(train.columns):
#    tscore[i],_=ttest_ind(train.loc[target==1,c].values,train.loc[target==0,c].values)
#tscore=pd.DataFrame(np.abs(tscore),index=train.columns)

#oe=OrdinalEncoder()
#cp_features=[]
#for c in features:
#    train[c+'_cp']=np.zeros(len(train))
#    test[c+'_cp']=np.zeros(len(test))
#    cp_features += [c+'_cp']
#train[cp_features]=oe.fit_transform(train[features])
#test[cp_features]=oe.transform(test[features])
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
used_features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]

ss=StandardScaler()
train=pd.DataFrame(ss.fit_transform(train[used_features]),columns=used_features)
test=pd.DataFrame(ss.transform(test[used_features]),columns=used_features)

folds = StratifiedKFold(n_splits=15, shuffle=False, random_state=4590)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

#LOO_list=['ProductName','SmartScreen','CityIdentifier','OsBuildLab','regionIdentifier','Platform','Processor','OsPlatformSubRelease','SkuEdition','Census_ProcessorClass','Census_PrimaryDiskTypeName','Census_ProcessorManufacturerIdentifier','Census_OSWUAutoUpdateOptionsName','Census_ActivationChannel','Census_MDC2FormFactor','Census_DeviceFamily','Census_FlightRing','Census_OSArchitecture','SmartScreen','Census_PowerPlatformRoleName','Census_OSBranch','Census_OSSkuName','Census_OSInstallTypeName','EngineVersion_3']
#LOO=LeaveOneOutEncoder(cols=LOO_list,randomized=True,handle_unknown='ignore')
#LOO.fit(train[features+['MachineIdentifier']],target)
#test=LOO.transform(test[features+['MachineIdentifier']])
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))
    train_x=train.iloc[trn_idx].reset_index(drop=True)
    valid_x=train.iloc[val_idx].reset_index(drop=True)
    target_train=target.iloc[trn_idx].reset_index(drop=True)
    target_valid=target.iloc[val_idx].reset_index(drop=True)
#    LOO=LeaveOneOutEncoder(cols=LOO_list,randomized=True,handle_unknown='ignore')
#    train_x=LOO.fit_transform(train_x[features],target_train)
#    valid_x=LOO.transform(valid_x[features])  
    model=LogisticRegression(solver='lbfgs', max_iter=15000, C=10)
#    model=SGDClassifier(loss='log', penalty='l2',
#                        alpha=0.000001,
##                        l1_ratio=0.3 ,
#                        max_iter=10000 ,tol=1e-3,  shuffle=True, verbose=10, n_jobs=-1, random_state=4590, learning_rate='constant', eta0=0.00001, power_t=0.5, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5)    
    model.fit(train_x[used_features],target_train)
    
    
    oof[val_idx] = model.predict_proba(valid_x[used_features])[:,1]
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = used_features
    fold_importance_df["importance"] = model.coef_.reshape(-1,1)
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += model.predict_proba(test[used_features])[:,1] / folds.n_splits
#============================================================
#    N = 5
#    p_valid,yp = 0,0
#    for i in range(N):
#        X_t, y_t = augment(train_x[used_features].values, target_train.values)
#        X_t = pd.DataFrame(X_t)
#        X_t = X_t.add_prefix('var_')
#        model=LogisticRegression(solver='lbfgs', max_iter=15000, C=10)
#        model.fit(X_t,y_t)
#    
#        oof[val_idx] += model.predict_proba(valid_x[used_features])[:,1]/N
#        predictions += model.predict_proba(test[used_features])[:,1] / (folds.n_splits*N)
print("CV score: {:<8.5f}".format(roc_auc_score(target,oof)))

#f_noimp_avg = (feature_importance_df[["feature", "importance"]]
#        .groupby("feature")
#        .mean()
#        .sort_values(by="importance", ascending=False))
#f=plt.figure(figsize=(12,5))
#plt.plot(f_noimp_avg)
#plt.xticks(rotation=-45)
#
#used_features=f_noimp_avg.reset_index()
#used_features=used_features.loc[used_features.loc[:,'importance']!=0,'feature'].tolist()
#f_noimp_avg.reset_index().to_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sub_mylogist_moreextref_fimp.csv', index=False)
df_test = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = predictions
df_test.to_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sub_mylogist_extref_15cv_{}_test.csv'.format(roc_auc_score(target,oof)), index=False)

df_train= pd.DataFrame({"ID_code":ID_code})
df_train['M_log']=oof
df_train.to_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sub_mylogist_extref_15cv_train.csv', index=False)

#f_noimp_avg=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mylogist_fimp.csv')
