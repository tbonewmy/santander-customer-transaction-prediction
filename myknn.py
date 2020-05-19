# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:18:29 2019

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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.impute import SimpleImputer
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import gc
from sklearn.model_selection import train_test_split
from scipy.stats import norm, rankdata

test = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/test.csv')
train = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/train.csv')
target=train['target']
features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]
ID_code=train['ID_code'].values
#a,_=train_test_split(train,train_size =0.2,shuffle =True,stratify =target ,random_state =4950)
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
used_features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]

ss=StandardScaler()
train=pd.DataFrame(ss.fit_transform(train[used_features]),columns=used_features)
test=pd.DataFrame(ss.transform(test[used_features]),columns=used_features)

folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=4590)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()
#t0 = time.time()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))    
    train_x=train.iloc[trn_idx].reset_index(drop=True)
    valid_x=train.iloc[val_idx].reset_index(drop=True)
    target_train=target.iloc[trn_idx].reset_index(drop=True)
    target_valid=target.iloc[val_idx].reset_index(drop=True)
#    model = KNeighborsClassifier(n_neighbors=100, leaf_size=3000, p=2, n_jobs=-1)
#    model.fit(train_x[features],target_train)
#    
#    oof[val_idx] = model.predict_proba(valid_x[features])[:,1]
#    
#    predictions += model.predict_proba(test[features])[:,1] / folds.n_splits
    #===============================
    N = 5
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(train_x[used_features].values, target_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
        model = KNeighborsClassifier(n_neighbors=200, leaf_size=6000, p=2, n_jobs=-1)
        model.fit(X_t,y_t)
    
        oof[val_idx] += model.predict_proba(valid_x[used_features])[:,1]/N
        predictions += model.predict_proba(test[used_features])[:,1] / (folds.n_splits*N)

print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#t1 = time.time()
#print('time: {}'.format(t1-t0))
df_test = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = predictions
df_test.to_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sub_myknn_aug_moref_15cv_200neigb_{}_test.csv'.format(roc_auc_score(target,oof)), index=False)

df_train= pd.DataFrame({"ID_code":ID_code})
df_train['M_knn']=oof
df_train.to_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sub_myknn_aug_moref_15cv_200neigb_train.csv', index=False)
