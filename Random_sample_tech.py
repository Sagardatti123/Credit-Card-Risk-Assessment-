''' here we implement random sample technique'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from log_code import setup_logging
logger=setup_logging('Random_sample_tech')

def clean_null_values(X_train,X_test):
    try:
        logger.info(f'train data before applying tech:\n{X_train.isnull().sum()}')
        logger.info(f'test data before applying tech:\n{X_test.isnull().sum()}')
        X_train['NumberOfDependents']=pd.to_numeric(X_train['NumberOfDependents'])
        X_test['NumberOfDependents']=pd.to_numeric(X_test['NumberOfDependents'])
        #random sample imputation

        for i in X_train.columns:
            if X_train[i].isnull().sum()>0:
                X_train[i+'_replaced']=X_train[i].copy()
                X_test[i+'_replaced'] = X_test[i].copy()
                s=X_train[i].dropna().sample(X_train[i].isnull().sum(),random_state=42)
                s1=X_test[i].dropna().sample(X_test[i].isnull().sum(),random_state=42)
                s.index=X_train[X_train[i].isnull()].index
                s1.index = X_test[X_test[i].isnull()].index
                X_train.loc[X_train[i].isnull(),i+'_replaced']=s
                X_test.loc[X_test[i].isnull(), i + '_replaced'] =s1
                X_train=X_train.drop([i],axis=1)
                X_test=X_test.drop([i],axis=1)
        logger.info('=====================')
        logger.info(f'after tech:\n {X_train.isnull().sum()}')
        logger.info(f'after tech:\n {X_test.isnull().sum()}')

        return X_train,X_test
        #logger.info(f'{X_train.info()}')
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')
