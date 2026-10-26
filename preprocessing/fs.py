import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
from scipy.stats import pearsonr
from  log_code import setup_logging
logger=setup_logging('fs')

from sklearn.feature_selection import VarianceThreshold

def select_best_columns(X_train,X_test,y_train,y_test):
    try:
        logger.info(f'before constant_tech X_train columns:{X_train.columns}:{X_train.shape}')
        logger.info(f'before constant_tech X_test columns:{X_test.columns}:{X_test.shape}')

        #constant technique
        cons_obj=VarianceThreshold(threshold=0.0)
        cons_obj.fit(X_train)
        logger.info(f'columns to remove after constant:{X_train.columns[~cons_obj.get_support()]}')
        X_train=X_train.drop(['NPA Status_yeo_trimming',
       'NumberOfTime30-59DaysPastDueNotWorse_yeo_trimming',
       'NumberOfTimes90DaysLate_yeo_trimming',
       'NumberOfTime60-89DaysPastDueNotWorse_yeo_trimming'],axis=1)
        X_test=X_test.drop(['NPA Status_yeo_trimming',
       'NumberOfTime30-59DaysPastDueNotWorse_yeo_trimming',
       'NumberOfTimes90DaysLate_yeo_trimming',
       'NumberOfTime60-89DaysPastDueNotWorse_yeo_trimming'],axis=1)
        logger.info(f'after constant_tech X_train columns:{X_train.columns}:{X_train.shape}')
        logger.info(f'after constant_tech X_test columns:{X_test.columns}:{X_test.shape}')

        #qusi constant
        qusi_obj=VarianceThreshold(threshold=0.1)
        qusi_obj.fit(X_train)
        logger.info(f'columns to remove after qusi_constant:{X_train.columns[~qusi_obj.get_support()]}')
        X_train=X_train.drop(['RevolvingUtilizationOfUnsecuredLines_yeo_trimming', 'NumberOfDependents_replaced_yeo_trimming'],axis=1)
        X_test=X_test.drop(['RevolvingUtilizationOfUnsecuredLines_yeo_trimming', 'NumberOfDependents_replaced_yeo_trimming'],axis=1)
        logger.info(f'after qusi_constant_tech X_train columns:{X_train.columns}:{X_train.shape}')
        logger.info(f'after qusi_constant_tech X_test columns:{X_test.columns}:{X_test.shape}')


        #correlation with hypothsies testing

        # corr_value=[]
        # for i in X_train.columns:
        #     values=pearsonr(X_train[i],y_train)
        #     corr_value.append(values)
        #     corr_value=np.array(corr_value)
        #     pvalue=corr_value[:,1]
        #     plt.figure(figsize=(5,3))
        #     plt.title('hyp testing')
        #     plt.xlabel('columns names')
        #     plt.ylabel('p vales of indepn columns')
        #     plt.bar(X_train.columns,pvalue)
        #     plt.show()
        X_train=X_train.drop(['DebtRatio_yeo_trimming'],axis=1)
        X_test=X_test.drop(['DebtRatio_yeo_trimming'],axis=1)
        logger.info(f'after hyp_testing_tech X_train columns:{X_train.columns}:{X_train.shape}')
        logger.info(f'after hyp_testing_tech X_test columns:{X_test.columns}:{X_test.shape}')
        return X_train,X_test
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')



