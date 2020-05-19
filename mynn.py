# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:08:55 2019

@author: wmy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:41:45 2019

@author: wmy
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import callbacks
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
from scipy.stats import norm, rankdata

class printAUC(callbacks.Callback):
    def __init__(self, X_train, y_train):
        super(printAUC, self).__init__()
        self.bestAUC = 0
        self.X_train = X_train
        self.y_train = y_train
        
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(np.array(self.X_train))
        auc = roc_auc_score(self.y_train, pred)
        print("Train AUC: " + str(auc))
        pred = self.model.predict(self.validation_data[0])
        auc = roc_auc_score(self.validation_data[1], pred)
        print ("Validation AUC: " + str(auc))
        if (self.bestAUC < auc) :
            self.bestAUC = auc
            self.model.save("/gpfs/home/mw15m/santander-customer-transaction-prediction/bestNet.h5", overwrite=True)
        return
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
test = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/test.csv')
train = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/train.csv')
target=train['target']
train_id=train['ID_code'].values
features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]
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
features=[c for c in train.columns.tolist() if c not in ['ID_code', 'target']]

#ss=StandardScaler()
#train=pd.DataFrame(ss.fit_transform(train[features]),columns=features)
#test=pd.DataFrame(ss.transform(test[features]),columns=features)

mms=MinMaxScaler()
train=pd.DataFrame(mms.fit_transform(train[features]),columns=features)
test=pd.DataFrame(mms.transform(test[features]),columns=features)

##SPLIT TRAIN AND VALIDATION SET
#X_train, X_val, Y_train, Y_val = train_test_split(
#    train[features], target,shuffle=True, stratify =target,test_size = 1/15, random_state=4590)
#X_train.reset_index(drop=True,inplace=True)
#X_val.reset_index(drop=True,inplace=True)
#Y_train.reset_index(drop=True,inplace=True)
#Y_val.reset_index(drop=True,inplace=True)
#x = gc.collect()
folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=4590)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))
    train_x=train.iloc[trn_idx].reset_index(drop=True)
    valid_x=train.iloc[val_idx].reset_index(drop=True)
    target_train=target.iloc[trn_idx].reset_index(drop=True)
    target_valid=target.iloc[val_idx].reset_index(drop=True)
    # BUILD MODEL
    model = Sequential()
    model.add(Dense(32,input_dim=len(features)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dense(100))
    #model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dense(100))
    #model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    #annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(target_train),
                                                     target_train)
    # TRAIN MODEL
    #def datagenxy(x,y,chunksize):
    #    ch=len(x)//chunksize
    #    for i in range(ch):
    #        yield x.iloc[i*chunksize:np.min([(i+1)*chunksize,len(x)]),:],y.iloc[i*chunksize:np.min([(i+1)*chunksize,len(x)])]
    model.fit(train_x,target_train, batch_size=20, epochs = 100, 
              callbacks=[
    #                  annealer,
              printAUC(train_x, target_train),EarlyStopping(monitor='val_loss',patience=10)],class_weight=class_weights, validation_data = (valid_x,target_valid), verbose=2)
    oof[val_idx]=model.predict(valid_x).flatten()
    
    predictions += model.predict(test[features]).flatten()/ folds.n_splits
    #=======================================================
#    N = 5
#    p_valid,yp = 0,0
#    for i in range(N):
#        model = Sequential()
#        model.add(Dense(32,input_dim=len(features)))
#        model.add(Dropout(0.5))
#        model.add(BatchNormalization())
#        model.add(Activation('relu'))
#        model.add(Dense(16))
#        model.add(Dropout(0.5))
#        model.add(BatchNormalization())
#        model.add(Activation('relu'))
#        #model.add(Dense(100))
#        #model.add(Dropout(0.4))
#        #model.add(BatchNormalization())
#        #model.add(Activation('relu'))
#        #model.add(Dense(100))
#        #model.add(Dropout(0.4))
#        #model.add(BatchNormalization())
#        #model.add(Activation('relu'))
#        model.add(Dense(1, activation='sigmoid'))
#        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
#        #annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)
#        
#        class_weights = class_weight.compute_class_weight('balanced',
#                                                         np.unique(target_train),
#                                                         target_train)
#        X_t, y_t = augment(train_x[features].values, target_train.values)
#        X_t = pd.DataFrame(X_t)
#        X_t = X_t.add_prefix('var_')
#        # TRAIN MODEL
#        #def datagenxy(x,y,chunksize):
#        #    ch=len(x)//chunksize
#        #    for i in range(ch):
#        #        yield x.iloc[i*chunksize:np.min([(i+1)*chunksize,len(x)]),:],y.iloc[i*chunksize:np.min([(i+1)*chunksize,len(x)])]
#        model.fit(X_t,y_t, batch_size=20, epochs = 100, 
#                  callbacks=[
#        #                  annealer,
#                  printAUC(train_x, target_train),EarlyStopping(monitor='val_loss',patience=10)],class_weight=class_weights, validation_data = (valid_x,target_valid), verbose=2)
#        oof[val_idx]=model.predict(valid_x).flatten()/N
#        
#        predictions += model.predict(test[features]).flatten()/ (folds.n_splits*N)

    #model.fit_generator(datagenxy(X_train,Y_train,2000000), steps_per_epoch=np.ceil(len(X_train)/2000000),epochs = 20, callbacks=[annealer,
    #          printAUC(X_train, Y_train)], validation_data = (X_val,Y_val), verbose=2)
    #model = load_model('./Kaggle/santander-customer-transaction-prediction/bestNet.h5')
    #./Kaggle/santander-customer-transaction-prediction/
    #del X_train, X_val, Y_train, Y_val
    #def datagenx(x,chunksize):
    #    ch=len(x)//chunksize
    #    for i in range(ch):
    #        yield x.iloc[i*chunksize:np.min([(i+1)*chunksize,len(x)]),:]
print("CV score: {:<8.5f}".format(roc_auc_score(target,oof)))
    #oof=np.zeros(len(train))
    #oof=model.predict_generator(datagenx(df_train[cols],2000000), steps=np.ceil(len(df_train)/2000000),verbose=10)  
    #oof=model.predict(df_train[cols])
    
    #ch=len(train)//2000000
    #for i in range(ch+1):
    #    print(i)
    #    upperbound=np.min([(i+1)*2000000,len(train)])
    #    temp_train=train.loc[i*2000000:upperbound-1,features]
    #    oof[i*2000000:upperbound]=model.predict_on_batch(temp_train).flatten()

#del train
#del temp_train
#x = gc.collect()

# LOAD BEST SAVED NET

#model = load_model('./Kaggle/santander-customer-transaction-prediction/bestNet.h5')

#pred = np.zeros((200000,1))
#id = 1

#chunksize = 2000000
#for df_test in pd.read_csv('./Kaggle/santander-customer-transaction-prediction/test.csv', 
#            chunksize = chunksize, usecols=list(dtypes.keys())[0:-1], dtype=dtypes):
#    print ('Loaded',len(df_test),'rows of TEST.CSV!')
#    # ENCODE TEST
#    cols = []
#    for x in FE:
#        cols += encode_FE(df_test,x,verbose=0)
#    for x in range(len(OHE)):
#        cols += encode_OHE_test(df_test,OHE[x],dd[x])
#    # PREDICT TEST
#    end = (id)*chunksize
#    if end>7853253: end = 7853253
#    pred[(id-1)*chunksize:end] = model.predict(df_test[cols])
#    print('  encoded and predicted part',id)
#    id += 1
df_test = pd.read_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sample_submission.csv')
df_test['target'] = predictions
df_test.to_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sub_mynn_extraf_15cv_0.5drop_32dense_adam_{}_test.csv'.format(roc_auc_score(target,oof)), index=False)
pre_train= pd.DataFrame({"ID_code":train_id})
pre_train['M_nn']=oof
pre_train.to_csv('/gpfs/home/mw15m/santander-customer-transaction-prediction/sub_mynn_extraf_15cv_0.5drop_32dense_adam_train.csv', index=False)
