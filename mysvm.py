# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:05:38 2019

@author: wmy
"""

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
#        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


DATA_TRAIN_PATH = './Kaggle/santander-customer-transaction-prediction/train.csv'
DATA_TEST_PATH = './Kaggle/santander-customer-transaction-prediction/test.csv'

def load_data(path_train = DATA_TRAIN_PATH, path_test = DATA_TEST_PATH):

    train = pd.read_csv(path_train)
    len_train = train.shape[0]
    print('\n Shape of train data:', train.shape)
    train_labels = train['target'].values
    train_ids = train['ID_code'].values
    features = train.columns[train.columns.str.startswith('var')].tolist()

    test = pd.read_csv(path_test)
    print(' Shape of test data:', test.shape)
    test_ids = test['ID_code'].values

    scaled, scaler = scale_data(np.concatenate((train[features].values, test[features].values), axis=0))
    train[features] = scaled[:len_train]
    test[features] = scaled[len_train:]

    train = train.drop(['target', 'ID_code'], axis=1).values
    test = test.drop(['ID_code'], axis=1).values

    return train, train_labels, test, train_ids, test_ids

starttime=timer(None)
X_train, y_train, X_test, tr_ids, te_ids = load_data()
seq_ids = np.arange(tr_ids.shape[0])
timer(starttime)

folds = 15
cv_sum = 0
pred = []
fpred = []

avreal = y_train
avpred = np.zeros(X_train.shape[0])
idpred = tr_ids

train_time=timer(None)
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=101)
for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    start_time=timer(None)
    print('\n Fold %02d' % (i+1))
    Xtrain, Xval = X_train[train_index], X_train[val_index]
    ytrain, yval = y_train[train_index], y_train[val_index]
    model = LinearSVC(C=0.01, tol=0.0001, verbose=1, random_state=1001, max_iter=2000, dual=False)
    isotonic = CalibratedClassifierCV(model, cv=5, method='isotonic')
    sigmoid = CalibratedClassifierCV(model, cv=5, method='sigmoid')

#    isotonic.fit(Xtrain, ytrain)
    sigmoid.fit(Xtrain, ytrain)
#    scores_val = isotonic.predict_proba(Xval)[:, 1]
    scores_val = sigmoid.predict_proba(Xval)[:, 1]
    ROC_AUC = roc_auc_score(yval, scores_val)
    print(' Fold %02d AUC: %.6f' % ((i+1), ROC_AUC))
#    y_pred = isotonic.predict_proba(X_test)[:, 1]
    y_pred = sigmoid.predict_proba(X_test)[:, 1]
    timer(start_time)

    avpred[val_index] = scores_val

    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    pred = fpred
    cv_sum = cv_sum + ROC_AUC

timer(train_time)

cv_score = (cv_sum / folds)
oof_ROC_AUC = roc_auc_score(avreal, avpred)
print('\n Average AUC: %.6f' % cv_score)
print(' Out-of-fold AUC: %.6f' % oof_ROC_AUC)

mpred = pred / folds

df_test = pd.read_csv('./Kaggle/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = mpred
df_test.to_csv('./Kaggle/santander-customer-transaction-prediction/sub_mysvm_15cv_{}_test.csv'.format(roc_auc_score(avreal, avpred)), index=False)

df_train= pd.DataFrame({"ID_code":tr_ids})
df_train['M_svm']=avpred
df_train.to_csv('./Kaggle/santander-customer-transaction-prediction/sub_mysvm_15cv_train.csv', index=False)
