# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:15:12 2019

@author: wmy
"""
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import SCORERS
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

lgbm_test=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mylgbm_aug_400bin_15cv_0.9019063722201077.csv')
lgbm_train=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mylgbm_aug_400bin_15cv_train.csv')
#fs_othermyLGBM1=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/feat_imp_othermyLGBM1.csv')
logit_train=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mylogist_extref_10cv_train.csv')
logit_test=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mylogist_extref_10cv_0.8967267941754058_test.csv')
#fs_otherLGBM=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/feat_imp_otherLGBM1.csv')
nn_train=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mynn_extraf_15cv_0.5drop_32dense_adam_train.csv')
nn_test=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mynn_extraf_15cv_0.5drop_32dense_adam_0.8627845574505735_test.csv')
#fs_mylgbm2=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/feat_imp_mylgbm2.csv')
knn_train=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_myknn_15cv_100neigb_train.csv')
knn_test=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_myknn_15cv_100neigb_0.7292868088632047_test.csv')
#mylgbm2_train.rename(columns={'M_my':'M_my2'},inplace=True)
#mylgbm2_test.rename(columns={'M_my':'M_my2'},inplace=True)
#fs_mylgbm1=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/feat_imp_mylgbm1.csv')
hubar_train=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_myhuber_15cv_train.csv')
hubar_test=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_myhuber_15cv_0.8548931246386763_test.csv')
#mylgbm1_train.rename(columns={'M_my':'M_my1'},inplace=True)
#mylgbm1_test.rename(columns={'M_my':'M_my1'},inplace=True)
svm_train=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mysvm_15cv_train.csv')
svm_test=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sub_mysvm_15cv_0.8596445022860983_test.csv')
#fs_othermyLGBM2=pd.read_csv('./Kaggle/santander-customer-transaction-prediction/feat_imp_othermyLGBM2.csv')
#f=plt.figure(figsize=(10,5))
#plt.plot(fs_othermyLGBM1.iloc[:10,1])
#plt.xticks(rotation=45)
#plt.ylabel('split gain')
#plt.tight_layout()

train = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/test.csv')
x_var=[c for c in train.columns.tolist() if c not in ['ID_code','target']]
target=train['target']

#from sklearn.preprocessing import PowerTransformer
#pt=PowerTransformer(method='yeo-johnson', standardize=False)
#train[x_var]=pt.fit_transform(train[x_var])
#test[x_var]=pt.transform(test[x_var])
#f, axes = plt.subplots(2, 200, figsize=(3.5*200, 3.5))
#for i in range(2):
#    for j in range(200):
#        sns.distplot(train.loc[target==i,x_var[j]] ,  ax=axes[i, j])
#plt.tight_layout()
#plt.savefig('./Kaggle/santander-customer-transaction-prediction/pair_boxcox.png')

#pg=sns.PairGrid(train,x_vars=x_var)
#sns.distplot()
#pg.map_offdiag(sns.distplot).savefig('./Kaggle/santander-customer-transaction-prediction/pair.png')
#FS-------------
#features_interest=[c for c in train.columns if c not in ['MachineIdentifier','HasDetections']]
#none_missing=(train[features_interest].isnull().sum()+test[features_interest].isnull().sum()).reset_index()
#none_missing=none_missing.loc[none_missing.iloc[:,1]==0,'index'].tolist()
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
#imp_features=pd.concat([fs_othermyLGBM1.loc[:50,'feature'],
#                        fs_otherLGBM.loc[:50,'feature'],
#                        fs_mylgbm2.loc[:50,'feature'],
#                        fs_mylgbm1.loc[:50,'feature'],
#                        fs_othermyLGBM2.loc[:50,'feature']],axis=0)
#imp_features=imp_features.unique()
#extra_features=np.setdiff1d(none_missing,imp_features).tolist()

train_block=pd.concat([
#        lgbm_train.iloc[:,1],
                       logit_train.iloc[:,1],
                       nn_train.iloc[:,1],
#                       svm_train.iloc[:,1],
                       knn_train.iloc[:,1]],axis=1)
test_block=pd.concat([
#        lgbm_test.iloc[:,1],
                       logit_test.iloc[:,1],
                       nn_test.iloc[:,1],
#                       svm_test.iloc[:,1],
                       knn_test.iloc[:,1]],axis=1)
test_block.columns=train_block.columns
a=np.corrcoef(train_block.values.T)
train_block=pd.concat([train_block,train],axis=1)
test_block=pd.concat([test_block,test],axis=1)
#test_block.columns=train_block.columns
#for c in extra_features:
#    le=LabelEncoder()
#    le.fit(np.unique(train_block[c].unique().tolist()+test_block[c].unique().tolist()))
#    train_block[c]=le.transform(train_block[c])
#    test_block[c]=le.transform(test_block[c])

features=[c for c in train_block.columns.tolist() if c not in ['ID_code','target']]

#Logistic
ss=StandardScaler()
train_block_ss=pd.DataFrame(ss.fit_transform(train_block[features]),columns=features)
test_block_ss=pd.DataFrame(ss.transform(test_block[features]),columns=features)
#param_C=np.logspace(-10,10,num=50)
#param_grid={'C':param_C}
folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=4590)
oof = np.zeros(len(train_block_ss))
predictions = np.zeros(len(test_block_ss))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_block_ss,target)):
    print("fold n°{}".format(fold_))    
    
    model=LogisticRegression(C=100,random_state=4590,solver='lbfgs',n_jobs=-1,verbose=10)    
    model.fit(train_block_ss.iloc[trn_idx][features],target.iloc[trn_idx])
    
    oof[val_idx] = model.predict_proba(train_block_ss.iloc[val_idx][features])[:,1]
    
    predictions += model.predict_proba(test_block_ss[features])[:,1] / folds.n_splits
print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#pre_test=pip_model.predict_proba(test_block[features])

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['HasDetections'] = predictions
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/stack_logist_C100.csv', index=False)

#SGDClassifier_logist
folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=4590)
oof = np.zeros(len(train_block_ss))
predictions = np.zeros(len(test_block_ss))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_block_ss,target)):
    print("fold n°{}".format(fold_))    

    model=SGDClassifier(loss='log', penalty='l2',alpha=0.00001, max_iter=1000 ,tol=1e-3,  shuffle=True, verbose=10, n_jobs=-1, random_state=4590, learning_rate='constant', eta0=0.0001, power_t=0.5, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5)    
    model.fit(train_block_ss.iloc[trn_idx][features],target.iloc[trn_idx])
    
    oof[val_idx] = model.predict_proba(train_block_ss.iloc[val_idx][features])[:,1]
    
    predictions += model.predict_proba(test_block_ss[features])[:,1] / folds.n_splits
print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#pre_test=pip_model.predict_proba(test_block[features])

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['HasDetections'] = predictions
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/stack_SGDClogist0.7417012794842577.csv', index=False)

#naive bay
folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=4590)
oof = np.zeros(len(train_block))
predictions = np.zeros(len(test_block))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_block,target)):
    print("fold n°{}".format(fold_))    

    model=GaussianNB(priors=None, var_smoothing=1e1)
    
    model.fit(train_block.iloc[trn_idx][features],target.iloc[trn_idx])
    
    oof[val_idx] = model.predict_proba(train_block.iloc[val_idx][features])[:,1]
    
    predictions += model.predict_proba(test_block[features])[:,1] / folds.n_splits
print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#pre_test=pip_model.predict_proba(test_block[features])

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['HasDetections'] = predictions
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/stack_SGDClogist0.7417012794842577.csv', index=False)

##SGDClassifier_perceptron
#folds = KFold(n_splits=5, shuffle=True, random_state=4590)
#oof = np.zeros(len(train_block_ss))
#predictions = np.zeros(len(test_block_ss))
#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_block_ss)):
#    print("fold n°{}".format(fold_))    
#
#    model=SGDClassifier(loss='perceptron', penalty='l2',alpha=0.3, max_iter=1000 ,tol=1e-3,  shuffle=True, verbose=10, n_jobs=-1, random_state=4590, learning_rate='constant', eta0=0.001, power_t=0.5, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5)    
#    model.fit(train_block_ss.iloc[trn_idx][features],target.iloc[trn_idx])
#    
#    oof[val_idx] = model.predict_proba(train_block_ss.iloc[val_idx][features])[:,1]
#    
#    predictions += model.predict_proba(test_block_ss[features])[:,1] / folds.n_splits
#print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#pre_test=pip_model.predict_proba(test_block[features])

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['HasDetections'] = predictions
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/stack_logist_C100.csv', index=False)

params = {'num_leaves': 343,#test
         'min_data_in_leaf': 296,
         'objective': 'binary',
         'max_depth': 15,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.038472676103018966,
         "bagging_freq": 5,
         "bagging_fraction": 0.47368907272124183,
         "bagging_seed": 11,
         "lambda_l1": 0.09827057924191607,
         "lambda_l2": 0.36962114333802165,
         "num_thread":-1,
         'metric':'auc',
         "random_state": 4950,          
         "verbosity": 10
         }
params = {#test
    'bagging_freq': 5,
#    'max_bin':400,
    'bagging_fraction': 0.8,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.004,
    'learning_rate': 0.001,
    'max_depth': 8,
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'num_threads': 5,
    'tree_learner': 'serial',
    "metric" : "auc",
    'objective': 'binary',
    'verbosity': 1}

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
folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=4590)
oof = np.zeros(len(train_block))
predictions = np.zeros(len(test_block))
#cor_matrix=np.zeros(5*5).reshape(5,5)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_block,target)):
    print("fold n°{}".format(fold_)) 
#    cor_matrix += np.corrcoef(train_block.iloc[val_idx,:].values.T)/folds.n_splits
    trn_data = lgb.Dataset(train_block.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
#                           categorical_feature=cat_cols
                          )
    val_data = lgb.Dataset(train_block.iloc[val_idx][features],
                           label=target.iloc[val_idx],
#                           categorical_feature=cat_cols
                          )

    num_round = 1000000
    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=5000,
                    early_stopping_rounds = 3000)
    
    oof[val_idx] = clf.predict(train_block.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    predictions += clf.predict(test_block[features], num_iteration=clf.best_iteration) / folds.n_splits
#============================================================
#    N = 3
#    p_valid,yp = 0,0
#    for i in range(N):
#        X_t, y_t = augment(train_block.iloc[trn_idx][features].values, target.iloc[trn_idx].values)
#        X_t = pd.DataFrame(X_t)
#        X_t = X_t.add_prefix('var_')
#    
#        trn_data = lgb.Dataset(X_t, label=y_t)
#        val_data = lgb.Dataset(train_block.iloc[val_idx][features], label=target.iloc[val_idx])
##        evals_result = {}
#        lgb_clf = lgb.train(params,
#                        trn_data,
#                        100000,
#                        valid_sets = [trn_data, val_data],
#                        early_stopping_rounds=3000,
#                        verbose_eval = 5000,
##                        evals_result=evals_result
#                       )
#        oof[val_idx] += lgb_clf.predict(train_block.iloc[val_idx][features], num_iteration=lgb_clf.best_iteration)/N
#        predictions += lgb_clf.predict(test_block[features], num_iteration=lgb_clf.best_iteration) / (folds.n_splits*N)

print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#pre_test=pip_model.predict_proba(test_block[features])

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = predictions
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/stack_lgmb_0.9001868721498363.csv', index=False)
#KNN
from sklearn.neighbors import KNeighborsClassifier
folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=4590)
oof = np.zeros(len(train_block_ss))
predictions = np.zeros(len(test_block_ss))
#t0 = time.time()
#model = KNeighborsClassifier(n_neighbors=100, leaf_size=3000, p=2, n_jobs=-1)
#model.fit(train_block_ss[features],target)
#oof = model.predict_proba(train_block_ss[features])[:,1]
#print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#
#predictions = model.predict_proba(test_block_ss[features])[:,1] 

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_block_ss,target)):
    print("fold n°{}".format(fold_))    

    model = KNeighborsClassifier(n_neighbors=20, leaf_size=3000, p=2, n_jobs=-1)
    model.fit(train_block_ss.iloc[trn_idx][features],target.iloc[trn_idx])
    
    oof[val_idx] = model.predict_proba(train_block_ss.iloc[val_idx][features])[:,1]
    
    predictions += model.predict_proba(test_block_ss[features])[:,1] / folds.n_splits
print('AUC Score: {}'.format(roc_auc_score(target,oof)))
#t1 = time.time()
print('time: {}'.format(t1-t0))
#pre_test=pip_model.predict_proba(test_block[features])

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = predictions
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/stack_knn100.csv', index=False)
#blend#
df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = (lgbm_test.iloc[:,1]*0.3+
                            logit_test.iloc[:,1]*0.3+
#                            nn_test.iloc[:,1]*0.1+
                            knn_test.iloc[:,1]*0.4)
#                            hubar_test.iloc[:,1]*0.1)
df_test['target'] = (lgbm_test.iloc[:,1]+
                            logit_test.iloc[:,1]+
                            nn_test.iloc[:,1]+
                            knn_test.iloc[:,1]+
                            svm_test.iloc[:,1]
                            )/5
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/blend_fifth.csv', index=False)

pd.concat([train_block,target],axis=1).to_csv('./Kaggle/santander-customer-transaction-prediction/train_block.csv', index=False)
