# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:36:49 2019

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
from sklearn.preprocessing import StandardScaler
import gc
import umap
from sklearn.decomposition import PCA
test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/test.csv')
train = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/train.csv')
target=train['target']
ID_code=train['ID_code'].values
features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]

exam=train[features]
exam=exam-exam.shift(1,axis=1)
exam=exam.dropna(axis=1)
f=plt.figure(figsize=(12,5))
plt.plot(exam.mean(axis=0).values)
plt.plot(exam.values.T)



describe=train[features].describe()

ss=StandardScaler()
train=pd.DataFrame(ss.fit_transform(train[features]),columns=features)
test=pd.DataFrame(ss.transform(test[features]),columns=features)

pca=PCA(n_components=2,random_state=4950)
pca.fit(train)
plt.plot(pca.explained_variance_ratio_)
train=pd.DataFrame(pca.transform(train),columns=['c1','c2'])
test=pd.DataFrame(pca.transform(test),columns=['c1','c2'])

f,ax=plt.subplots(int(np.ceil((train_cate.shape[1]/5))),5,figsize=(5*5,np.ceil((train_cate.shape[1]/5))*5))
features=[c for c in train_cate.columns.tolist() if c not in ['ID_code', 'target']]
for i in range(train_cate.shape[1]//5):
    for j in range(5):
#        sns.distplot(train_cate.loc[target==1,features[i*5+j]].values,ax=ax[i,j],label='1')
#        sns.distplot(train_cate.loc[target==0,features[i*5+j]].values,ax=ax[i,j],label='0')
        pd.DataFrame(train_cate.loc[target==1,features[i*5+j]]).plot(kind ='bar',legend =True)
        pd.DataFrame(train_cate.loc[target==0,features[i*5+j]]).plot(kind ='bar',legend =True)

#        ax[i,j].plot(train_cate[features[i*5+j]].values,target.values,'.')
        ax[i,j].set_title(features[i*5+j])
#        ax[i,j].legend(loc='best')
#        print(features[i*5+j])
plt.tight_layout()
plt.savefig('./Kaggle/santander-customer-transaction-prediction/distri.png')
from scipy.stats import gaussian_kde
describe1=train.loc[target==1,features].describe()
describe0=train.loc[target==0,features].describe()

num_point=100
col_name='var_5'
start=np.max([describe1.loc['min',col_name],describe0.loc['min',col_name]])
finish=np.max([describe1.loc['max',col_name],describe0.loc['max',col_name]])
line=np.linspace(start,finish,num_point)
g1=gaussian_kde(train.loc[target==1,col_name])
p1=g1.evaluate(line)
g0=gaussian_kde(train.loc[target==0,col_name])
p0=g0.evaluate(line)

#np.abs(p1-p0)
inter=[]
skip_num=1
max_point=np.mean([np.max(p1),np.max(p0)])
threshold=max_point*0.004
for i in range(len(p1)-skip_num):
    mag=(np.abs(p1[i]-p0[i])+np.abs(p1[i+skip_num]-p0[i+skip_num]))/2>threshold
    if p1[i]>p0[i]:
        if p1[i+skip_num]<p0[i+skip_num] and mag:
            inter.extend([(line[i+skip_num]+line[i])/2])
    else:
        if p1[i+skip_num]>p0[i+skip_num] and mag:
            inter.extend([(line[i+skip_num]+line[i])/2])
plt.plot(line,p1)
plt.plot(line,p0)
for i in inter:
    plt.axvline(i)
    
new_col=np.zeros(len(train))
for i in range(len(b)):
   new_col[train[col_name]>b[i]] =+ 1
#train[features]=train[features]-train[features].mean()
describe1=train.loc[target==1,features].describe().drop(['count','min','max'],axis=0)
describe0=train.loc[target==0,features].describe().drop(['count','min','max'],axis=0)
mean1=train.loc[target==1,features].mean()
mean0=train.loc[target==0,features].mean()
similar=np.abs(describe1-describe0)<0.05
feature_rank=similar.sum(axis=0).sort_values(ascending=True)
sum(feature_rank!=5)
plot_f=feature_rank[feature_rank==5].index.tolist()
used_features=feature_rank[feature_rank!=5].index.tolist()
f,ax=plt.subplots(5,5,figsize=(5*5,5*5))
k=0
for i in range(5):
    for j in range(5):
        sns.distplot(np.exp(train.loc[target==1,plot_f[k]]),ax=ax[i,j])
        sns.distplot(np.exp(train.loc[target==0,plot_f[k]]),ax=ax[i,j])
        plt.title(plot_f[k])
        if k==21:
            break
        k += 1
#        print(features[i*5+j])
plt.tight_layout()
plt.savefig('./Kaggle/santander-customer-transaction-prediction/similar_exp().png')
for c in plot_f:
    print(train[c].nunique())
    
um=umap.UMAP(n_neighbors=20,k=20)
embedding = um.fit_transform(train[features])
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(embedding[:,0],embedding[:,1],c=target, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
