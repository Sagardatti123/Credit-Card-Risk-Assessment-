from cProfile import label

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from hypothesis.internal import entropy

from log_code import setup_logging
logger =setup_logging('all_models')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve

def knn_algo(X_train,y_train,X_test,y_test):
    try:
        global knn_reg
        knn_reg=KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(X_train,y_train)
        logger.info(f'test accuracy of KNN:{accuracy_score(y_test,knn_reg.predict(X_test))}')
        logger.info(f'test confusion matrix KNN:\n:{confusion_matrix(y_test,knn_reg.predict(X_test))}')
        logger.info(f'test classification matrix KNN:\n:{classification_report(y_test,knn_reg.predict(X_test))}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')



def nb_algo(X_train,y_train,X_test,y_test):
    try:
        global nb_reg
        nb_reg=GaussianNB()
        nb_reg.fit(X_train,y_train)
        logger.info(f'test accuracy of NB:{accuracy_score(y_test,nb_reg.predict(X_test))}')
        logger.info(f'test confusion matric NB:\n:{confusion_matrix(y_test,nb_reg.predict(X_test))}')
        logger.info(f'test classification report NB:\n:{classification_report(y_test,nb_reg.predict((X_test)))}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


def lr_algo(X_train,y_train,X_test,y_test):
    try:
        global lr_reg
        lr_reg=LogisticRegression()
        lr_reg.fit(X_train,y_train)
        logger.info(f'test accuracy of LR:{accuracy_score(y_test,lr_reg.predict(X_test))}')
        logger.info((f'test confusion matric of LR:\n:{confusion_matrix(y_test,lr_reg.predict(X_test))}'))
        logger.info(f'test classification report of LR:\n:{classification_report(y_test,lr_reg.predict(X_test))}')

    except Exception as e:
        er_type,er_message,er_line=sys.exc_info()
        logger.info(f'error in the line:{er_line.tb_lineno} due to :{er_type} reason:{er_message}')

def dt_algo(X_train, y_train, X_test, y_test):
    try:
        global dt_reg
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(X_train, y_train)
        logger.info(f'test accuracy of DT:{accuracy_score(y_test, dt_reg.predict(X_test))}')
        logger.info(f'test confusion matric DT:\n:{confusion_matrix(y_test, dt_reg.predict(X_test))}')
        logger.info(f'test classification report DT:\n:{classification_report(y_test, dt_reg.predict((X_test)))}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


def rf_algo(X_train, y_train, X_test, y_test):
    try:
        global rf_reg
        rf_reg = RandomForestClassifier(criterion='entropy',n_estimators=10)
        rf_reg.fit(X_train, y_train)
        logger.info(f'test accuracy of RF:{accuracy_score(y_test, rf_reg.predict(X_test))}')
        logger.info(f'test confusion matric RF:\n:{confusion_matrix(y_test, rf_reg.predict(X_test))}')
        logger.info(f'test classification report RF:\n:{classification_report(y_test, rf_reg.predict((X_test)))}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')

def ab_algo(X_train, y_train, X_test, y_test):
    try:
        global ab_reg
        from sklearn.linear_model import   LogisticRegression
        lr=LogisticRegression()
        ab_reg = AdaBoostClassifier(estimator=lr,n_estimators=10)
        ab_reg.fit(X_train, y_train)
        logger.info(f'test accuracy of AB:{accuracy_score(y_test, ab_reg.predict(X_test))}')
        logger.info(f'test confusion matric AB:\n:{confusion_matrix(y_test, ab_reg.predict(X_test))}')
        logger.info(f'test classification report AB:\n:{classification_report(y_test, ab_reg.predict((X_test)))}')
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


def gb_algo(X_train, y_train, X_test, y_test):
    try:
        global gb_reg
        gb_reg = GradientBoostingClassifier(n_estimators=10)
        gb_reg.fit(X_train, y_train)
        logger.info(f'test accuracy of GB:{accuracy_score(y_test, gb_reg.predict(X_test))}')
        logger.info(f'test confusion matric GB:\n:{confusion_matrix(y_test, gb_reg.predict(X_test))}')
        logger.info(f'test classification report GB:\n:{classification_report(y_test, gb_reg.predict((X_test)))}')
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


def xgb_algo(X_train, y_train, X_test, y_test):
    try:
        global xgb_reg
        xgb_reg = XGBClassifier()
        xgb_reg.fit(X_train, y_train)
        logger.info(f'test accuracy of XGB:{accuracy_score(y_test, xgb_reg.predict(X_test))}')
        logger.info(f'test confusion matric XGB:\n:{confusion_matrix(y_test, xgb_reg.predict(X_test))}')
        logger.info(f'test classification report XGB:\n:{classification_report(y_test, xgb_reg.predict((X_test)))}')
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')


def auc_roc_curved_strct(X_train,y_train,X_test,y_test):
    try:
        knn_predictions=knn_reg.predict(X_test)
        nb_predictions = nb_reg.predict(X_test)
        lr_predictions = lr_reg.predict(X_test)
        dt_predictions = dt_reg.predict(X_test)
        rf_predictions = rf_reg.predict(X_test)
        ab_predictions = ab_reg.predict(X_test)
        gb_predictions = gb_reg.predict(X_test)
        xgb_predictions = xgb_reg.predict(X_test)


        knn_fp,knn_tp,knn_thre=roc_curve(y_test,knn_predictions)
        nb_fp, nb_tp, nb_thre = roc_curve(y_test, nb_predictions)
        lr_fp, lr_tp, lr_thre = roc_curve(y_test, lr_predictions)
        dt_fp, dt_tp, dt_thre = roc_curve(y_test, dt_predictions)
        rf_fp, rf_tp, rf_thre = roc_curve(y_test, rf_predictions)
        ab_fp, ab_tp, ab_thre = roc_curve(y_test, ab_predictions)
        gb_fp, gb_tp, gb_thre = roc_curve(y_test, gb_predictions)
        xgb_fp, xgb_tp, xgb_thre = roc_curve(y_test, xgb_predictions)

        plt.figure(figsize=(5,3))
        plt.title('AUC & ROC curve')
        plt.xlabel('flase positive rate')
        plt.ylabel('true positive rate')

        plt.plot(knn_fp,knn_tp,label='KNN')
        plt.plot(nb_fp, nb_tp, label='NB')
        plt.plot(lr_fp, lr_tp, label='LR')
        plt.plot(dt_fp, dt_tp, label='DT')
        plt.plot(rf_fp, rf_tp, label='RF')
        plt.plot(ab_fp, ab_tp, label='AB')
        plt.plot(gb_fp, gb_tp, label='GB')
        plt.plot(xgb_fp, xgb_tp, label='XGB')
        plt.legend(loc=0)
        plt.show()

        logger.info(f'gradient boosting is better AUC and RUC so choose GradientBoosting')






    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')










def common(X_train,y_train,X_test,y_test):
    try:
        logger.info('=============KNN============')
        knn_algo(X_train,y_train,X_test,y_test)

        logger.info('=============NB=============')
        nb_algo(X_train,y_train,X_test,y_test)

        logger.info('==============LR=============')
        lr_algo(X_train,y_train,X_test,y_test)

        logger.info('==============DT=============')
        dt_algo(X_train, y_train, X_test, y_test)

        logger.info('==============RF=============')
        rf_algo(X_train, y_train, X_test, y_test)

        logger.info('==============AB=============')
        ab_algo(X_train, y_train, X_test, y_test)

        logger.info('==============GB=============')
        gb_algo(X_train, y_train, X_test, y_test)

        logger.info('==============XGB=============')
        xgb_algo(X_train, y_train, X_test, y_test)

        logger.info('==============AUC & ROC============')
        auc_roc_curved_strct(X_train,y_train,X_test,y_test)


    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'error in the line :{er_line.tb_lineno} due to :{er_type} resaon:{er_mess}')

