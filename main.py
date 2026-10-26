'''  we find the credit card data'''
import warnings
#from unittest.mock import inplace

#from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os
import sys
import warnings

from pandas.conftest import axis

from fs import select_best_columns

#from narwhals import exclude
#from pandas.conftest import axis
warnings.filterwarnings('ignore')

from log_code import setup_logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder #converts categricol data to numerical data
from imblearn.over_sampling import SMOTE #smote is cause for oversampling tech (data balancing)
from sklearn.preprocessing import StandardScaler # z-score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle
logger=setup_logging('main')

from Random_sample_tech import clean_null_values
from yeojhonson_trimming import variable_tran_outliers
from fs import select_best_columns
from all_models import common

class CREDIT_CARD_DUE:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(f'the number of row and columns:{self.df.shape}')
            logger.info(f'calculate null values \n {self.df.isnull().sum()}')
            #logger.info(self.df.tail(2))
            self.df=self.df.drop([150000,150001],axis=0)
            self.df=self.df.drop(['MonthlyIncome.1'],axis=1)
            logger.info(f'calculate null values \n {self.df.isnull().sum()}')
            self.X=self.df.iloc[ : ,:-1]
            self.y=self.df.iloc[ : ,-1]
            self.y=self.y.map({'Good':1,'Bad':0}).astype(int)
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'training datasets size: \n{self.X_train.shape} == {self.y_train.shape}')
            logger.info(f'testing data size: \n{self.X_test.shape}== {self.y_test.shape}')
        except Exception as e:
            er_type,er_mess,er_line=sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


    def handling_missing_values(self):
        try:
            logger.info(f'before tech:{self.X_train.isnull().sum()}')
            logger.info(f'before tech:{self.X_test.isnull().sum()}')
            self.X_train,self.X_test=clean_null_values(self.X_train,self.X_test)
            logger.info(f'after tech:{self.X_train.isnull().sum()}')
            logger.info(f'after tech:{self.X_test.isnull().sum()}')

        except Exception as e:
            er_type,er_mess,er_line=sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')

        logger.info('============================================')

    def splitting_numreical_catgery(self):
        try:
            self.X_train_numerical=self.X_train.select_dtypes(exclude='object')
            self.X_train_catgery=self.X_train.select_dtypes(include='object')
            self.X_test_numerical=self.X_test.select_dtypes(exclude='object')
            self.X_test_catgery=self.X_test.select_dtypes(include='object')

            logger.info(f'train numerical columns:{self.X_train_numerical.shape} \n {self.X_train_numerical.columns}')
            logger.info(f'train categricol columns:{self.X_train_catgery.shape} \n {self.X_train_catgery.columns}')
            logger.info(f'test numerical columns:{self.X_test_numerical.shape} \n {self.X_test_numerical.columns}')
            logger.info(f'test categorical columns:{self.X_test_catgery.shape} \n {self.X_test_catgery.columns}')
        except Exception as e:
            er_type,er_mess,er_line=sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')

    def outliers(self):
        try:
            self.X_train_numerical,self.X_test_numerical=variable_tran_outliers(self.X_train_numerical,self.X_test_numerical)
        except Exception as e:
            er_type, er_mess, er_line = sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


    def feature_selection(self):
        try:
            self.X_train_numerical,self.X_test_numerical=select_best_columns(self.X_train_numerical,self.X_test_numerical,self.y_train,self.y_test)
        except Exception as e:
            er_type, er_mess, er_line = sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')
    def cat_to_num(self):
        try:
            logger.info(f'X_train categorical columns and values before tech:{self.X_train_catgery.columns}:\n:{self.X_train_catgery.head(10)}')
            logger.info(f'X_test categorical columns and values before tech:{self.X_test_catgery.columns}:\n:{self.X_test_catgery.head(10)}')
            #apply one hot encoding to nominal columns that are gender and region
            one_obj=OneHotEncoder(drop='first')
            one_obj.fit(self.X_train_catgery[['Gender','Region']])
            result1=one_obj.transform(self.X_train_catgery[['Gender','Region']])
            result2=one_obj.transform(self.X_test_catgery[['Gender','Region']])
            result1=result1.toarray()
            result2=result2.toarray()
            t1=pd.DataFrame(data=result1,columns=one_obj.get_feature_names_out())
            t2=pd.DataFrame(data=result2,columns=one_obj.get_feature_names_out())
            self.X_train_catgery.reset_index(drop=True,inplace=True)
            self.X_test_catgery.reset_index(drop=True,inplace=True)
            t1.reset_index(drop=True,inplace=True)
            t2.reset_index(drop=True,inplace=True)
            self.X_train_catgery=pd.concat([self.X_train_catgery,t1],axis=1)
            self.X_test_catgery=pd.concat([self.X_test_catgery,t2],axis=1)
            self.X_train_catgery=self.X_train_catgery.drop(['Gender','Region'],axis=1)
            self.X_test_catgery=self.X_test_catgery.drop(['Gender','Region'],axis=1)
            logger.info(
                f'X_train categorical columns and values after onehot tech:{self.X_train_catgery.columns}:\n:{self.X_train_catgery.head(10)}')
            logger.info(
                f'X_test categorical columns and values after onehot tech:{self.X_test_catgery.columns}:\n:{self.X_test_catgery.head(10)}')

            #odinal encoding for remining columns
            odinal_obj=OrdinalEncoder()
            odinal_obj.fit(self.X_train_catgery[['Rented_OwnHouse', 'Occupation', 'Education']])
            result3=odinal_obj.transform(self.X_train_catgery[['Rented_OwnHouse', 'Occupation', 'Education']])
            result4=odinal_obj.transform(self.X_test_catgery[['Rented_OwnHouse', 'Occupation', 'Education']])
            p1=pd.DataFrame(data=result3,columns=odinal_obj.get_feature_names_out()+'_res')
            p2=pd.DataFrame(data=result4,columns=odinal_obj.get_feature_names_out()+'_res')
            self.X_train_catgery.reset_index(drop=True,inplace=True)
            self.X_test_catgery.reset_index(drop=True,inplace=True)
            p1.reset_index(drop=True,inplace=True)
            p2.reset_index(drop=True,inplace=True)
            self.X_train_catgery=pd.concat([self.X_train_catgery,p1],axis=1)
            self.X_test_catgery=pd.concat([self.X_test_catgery,p2],axis=1)
            self.X_train_catgery=self.X_train_catgery.drop(['Rented_OwnHouse', 'Occupation', 'Education'],axis=1)
            self.X_test_catgery=self.X_test_catgery.drop(['Rented_OwnHouse', 'Occupation', 'Education'],axis=1)
            logger.info(
                f'X_train categorical columns and values after odinal tech:{self.X_train_catgery.columns}:\n:{self.X_train_catgery.head(10)}')
            logger.info(
                f'X_test categorical columns and values after odinal tech:{self.X_test_catgery.columns}:\n:{self.X_test_catgery.head(10)}')

            logger.info(f'train data:{self.X_train_catgery.isnull().sum()}')
            logger.info(f'test data:{self.X_test_catgery.isnull().sum()}')


            self.X_train_numerical.reset_index(drop=True,inplace=True)
            self.X_test_numerical.reset_index(drop=True,inplace=True)
            self.X_train_catgery.reset_index(drop=True,inplace=True)
            self.X_test_catgery.reset_index(drop=True,inplace=True)

            self.final_training_data=pd.concat([self.X_train_numerical,self.X_train_catgery],axis=1)
            self.final_testing_data=pd.concat([self.X_test_numerical,self.X_test_catgery],axis=1)

            logger.info('========================================')
            logger.info(f'==============final training data============:\n:{self.final_training_data.shape}:\n:{self.final_training_data.columns}:\n:{self.final_training_data.head(10)}:\n:{self.final_training_data.isnull().sum()}')
            logger.info(f'==============final testing data============:\n:{self.final_testing_data.shape}:\n:{self.final_testing_data.columns}:\n:{self.final_testing_data.head(10)}:\n:{self.final_testing_data.isnull().sum()}')
        except Exception as e:
            er_type, er_mess, er_line = sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')
    def data_balancing(self):
        try:
            logger.info(f'before balancing total final training data:{self.y_train.shape}')
            logger.info(f'before balancing no of rows for good:{1}:class :{sum(self.y_train==1)}')
            logger.info(f'before balancing no of rows for bad:{0}:class:{sum(self.y_train==0)}')
            sm_obj=SMOTE(random_state=42)
            self.X_train_balanced,self.y_train_balanced=sm_obj.fit_resample(self.final_training_data,self.y_train)
            logger.info(f'after balancing total final training data:{self.y_train_balanced.shape}')
            logger.info(f'after balancing no of rows for good:{1}:class:{sum(self.y_train_balanced==1)}')
            logger.info(f'after balancing no of rows for bad:{0}:class:{sum(self.y_train_balanced==0)}')


            #lets scale down the values

            sc=StandardScaler()
            sc.fit(self.X_train_balanced)
            self.X_train_balanced_scaled=sc.transform(self.X_train_balanced)
            self.X_test_scaled=sc.transform(self.final_testing_data)
            with open('scaled model.pkl','wb') as f:
                pickle.dump(sc,f)

        except Exception as e:
            er_type, er_mess, er_line = sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')

    # we can give the data to ml algorithms
    # training varibles:(self.X_train_balanced_scaled,self.y_train_balanced)

    def train_all_models(self):
        try:
            #common(self.X_train_balanced_scaled,self.y_train_balanced,self.X_test_scaled,self.y_test)
            #since above we know the model such as gredient boosting thats why we commont insted run every time
            gb_reg=GradientBoostingClassifier(n_estimators=10)
            gb_reg.fit(self.X_train_balanced_scaled,self.y_train_balanced)
            logger.info(f'test accuracy of GB:{accuracy_score(self.y_test,gb_reg.predict(self.X_test_scaled))}')
            logger.info(f'test confusion matrix of GB:\n:{confusion_matrix(self.y_test,gb_reg.predict(self.X_test_scaled))}')
            logger.info(f'test classification report GB:\n:{classification_report(self.y_test,gb_reg.predict(self.X_test_scaled))}')


            #save  the model as pickle file
            with open('creditcard_model.pkl','wb') as f1:
                pickle.dump(gb_reg,f1)


        except Exception as e:
            er_type, er_mess, er_line = sys.exc_info()
            logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


if __name__=="__main__":
    obj=CREDIT_CARD_DUE('creditcard.csv')
    obj.handling_missing_values()
    obj.splitting_numreical_catgery()
    obj.outliers()
    obj.feature_selection()
    obj.cat_to_num()
    obj.data_balancing()
    obj.train_all_models()

