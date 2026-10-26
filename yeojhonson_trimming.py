import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
import scipy
import seaborn as sns
from pandas.conftest import axis
from scipy.stats import yeojohnson
from log_code import setup_logging
logger=setup_logging('yeojhonson_trimming')

def variable_tran_outliers(X_train,X_test):
    try:
        logger.info(f'before X_train column names:{X_train.columns}')
        logger.info(f'before X_test columns names:{X_test.columns}')
        for i in X_train.columns:
            #apply yeo_jhonson to every columns retrive for normal distribution
            #and updated into columns_yeo name in x_train
            X_train[i+'_yeo'],lam_value=yeojohnson(X_train[i])
            X_test[i+'_yeo'],lam_value=yeojohnson(X_test[i])
            #drop that particular column and updated into x_train
            X_train=X_train.drop([i],axis=1)
            X_test=X_test.drop([i],axis=1)

            iqr=X_train[i+'_yeo'].quantile(0.75)-X_train[i+'_yeo'].quantile(0.25)
            upper=X_train[i+'_yeo'].quantile(0.75)+(1.5*iqr)
            lower=X_train[i+'_yeo'].quantile(0.25)-(1.5*iqr)

            X_train[i+'_yeo_trimming']=np.where(X_train[i+'_yeo']>upper,upper,
                                                np.where(X_train[i+'_yeo']<lower,lower,X_train[i+'_yeo']))
            X_test[i + '_yeo_trimming'] = np.where(X_test[i + '_yeo'] > upper, upper,
                                                    np.where(X_test[i + '_yeo'] < lower, lower, X_test[i + '_yeo']))


            X_train=X_train.drop([i+'_yeo'],axis=1)
            X_test=X_test.drop([i+'_yeo'],axis=1)

        logger.info(f'after X_train column names:{X_train.columns},{X_train.shape}')
        logger.info(f'after X_test columns names:{X_test.columns},{X_test.shape}')
        #logger.info(f'{X_train.shape}')
        return X_train,X_test
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')
