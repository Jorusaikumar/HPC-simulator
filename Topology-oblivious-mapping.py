import os
import cPickle as pickle
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import LeaveOneOut,cross_validate,KFold
from sklearn.feature_selection import SelectKBest,f_classif,SelectPercentile,RFE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class MLTrainer(object):
    
    def __init__(self):
        path=os.path.dirname(os.path.abspath(__file__))
        df=pd.read_csv(path+'/database/dataset.csv')
        for column in df.columns:
            if len(df[column].unique())==1:
                df.drop(column,inplace=True,axis=1)
        scaler=StandardScaler()
        scaler.fit(df.iloc[:,1:-2])
        df.iloc[:,1:-2]=scaler.transform(df.iloc[:,1:-2])
        self.df=df
        self.scaler=scaler
    
    def loadmodel(self,name,model_type):
        if model_type is 'classification':
            
            if name is 'XGBoost':
                return xgb.XGBClassifier(learning_rate=0.1,max_depth=15,n_estimators=200,objective='binary:logistic',
                                gamma=0.1,min_child_weight=1,colsample_bytree=0.3,silent=True)
            if name is 'LogisticRegression':
                return LogisticRegression(penalty='l2',C=0.012742749857031334,max_iter=100)
            if name is 'SVM':
                return svm.SVC()

        elif model_type is 'regression':
            
            if name is 'LinearRegression':
                return LinearRegression(fit_intercept=False)
            if name is 'SVM':
                return svm.SVR()
            if name is 'XGBoost':
                return xgb.XGBRegressor(colsample_bytree= 0.7, learning_rate= 0.05, min_child_weight= 3,
                                        n_estimators= 100, max_depth= 10, gamma= 0.3)
            
        elif model_type is 'multiclass':
            
            if name is 'XGBoost':
                return xgb.XGBClassifier(objective='multi:softmax',num_class=10,learning_rate=0.01,max_depth=7,colsample_bytree=0.8,
                                        n_estimators=300)
            
            if name is 'NeuralNetwork':
                model=Sequential()
                model.add()

        return None
    
    def MLClassifier(self,classifier):
        X,y=self.ImbalancedClassification()
        X=self.univariateFeatureSelection(X,y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        model=self.loadmodel(classifier,'classification')
        model.fit(X_train,Y_train)
        preds=model.predict(X_test)
        accuracy=accuracy_score(preds,Y_test)
        precision=precision_score(preds,Y_test)
        recall=recall_score(preds,Y_test)
        return model,accuracy,precision,recall
    
    def LeaveOneOutCrossValidationMeasure(self, classifier):
        X,y=self.ImbalancedClassification()
        X=self.univariateFeatureSelection(X,y)
        cv = LeaveOneOut()
        model=self.loadmodel(classifier,'classification')
        scores = cross_validate(model, X, y, scoring=['accuracy','precision','recall'], cv=cv, n_jobs=-1)
        return np.mean(scores['test_accuracy']),np.mean(scores['test_precision']),np.mean(scores['test_recall'])
    
    def nfoldcrossvalidation(self,n,classifier):
        X,y=self.ImbalancedClassification()
        X=self.univariateFeatureSelection(X,y)
        kf=KFold(n_splits=n,shuffle=True)
        tot_accuracy=0.0
        tot_precision=0.0
        tot_recall=0.0
        for train_index,test_index in kf.split(X):
            X_train,X_test=X[train_index],X[test_index]
            y_train,y_test=y[train_index],y[test_index]
            model=self.loadmodel(classifier,'classification')
            model.fit(X_train,y_train)
            preds=model.predict(X_test)
            tot_accuracy+=accuracy_score(preds,y_test)
            tot_precision+=precision_score(preds,y_test)
            tot_recall+=recall_score(preds,y_test)
        tot_accuracy/=n
        tot_precision/=n
        tot_recall/=n
        return tot_accuracy,tot_precision,tot_recall
    
    def univariateFeatureSelection(self,X,y):
        test=SelectKBest(score_func=f_classif,k=20)
        fit = test.fit(X, y)
        mask=fit.get_support()
        features=[]
        for B,F in zip(mask, list(self.df.columns)[1:-2]):
            if B:
                features.append(F)
        X_new=fit.transform(X)
        return X_new,features
    
    def recursiveFeatureElimination(self,classifier,X,y):
        model=self.loadmodel(classifier,'classification')
        rfe=RFE(model,10)
        fit=rfe.fit(X,y)
        features=fit.transform(X)
        return features
    
    def ImbalancedClassification(self):
        X=np.array(self.df.drop(['kernel','cputime','gputime'],axis=1))
        cpu=np.array(self.df['cputime'])
        gpu=np.array(self.df['gputime'])
        y=(cpu>gpu)
        oversample=SMOTE()
        X,y=oversample.fit_sample(X,y)
        return X,y
    
    def removekernels(self,L):
        print self.df.shape
        self.df=self.df.set_index("kernel").drop(L,axis=0)
        print self.df.shape
        
    def regression(self,M):
        X=np.array(self.df.drop(['kernel','cputime','gputime'],axis=1))
        cpu=np.array(self.df['cputime'])
        gpu=np.array(self.df['gputime'])
        y=cpu/gpu
        
        #filter out the examples in minor area where y>5
        del_indices=[idx for idx, val in enumerate(y) if val > 5]
        X=np.delete(X,del_indices,axis=0)
        y=np.delete(y,del_indices)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        model=self.loadmodel(M,'regression')
        model.fit(X_train,Y_train)
        preds=model.predict(X_test)
        rmse=sqrt(mean_squared_error(Y_test,preds))
        mae=mean_absolute_error(Y_test,preds)
        r2=r2_score(Y_test,preds)
        mape=self.mean_absolute_percentage_error(Y_test,preds)
        return model,mape,rmse,mae,r2
    
    def mean_absolute_percentage_error(self,y_true,y_pred):
        y_pred=np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def load_multiclass_data(self):
        feats=np.array(self.df.drop(['kernel','cputime','gputime'],axis=1))
        cpu=np.array(self.df['cputime'])
        gpu=np.array(self.df['gputime'])
        ratios=cpu/gpu
        
        del_indices=[idx for idx, val in enumerate(ratios) if val > 5]
        feats=np.delete(feats,del_indices,axis=0)
        ratios=np.delete(ratios,del_indices)
        y=[]
        feats=self.univariateFeatureSelection(feats,ratios)
        
        for R in ratios:
            if R<0.5:
                y.append(0)
            elif R<1.0:
                y.append(1)
            elif R<1.5:
                y.append(2)
            elif R<2.0:
                y.append(3)
            elif R<2.5:
                y.append(4)     
            elif R<3.0:
                y.append(5)
            elif R<3.5:
                y.append(6)
            elif R<4.0:
                y.append(7)
            elif R<4.5:
                y.append(8)
            elif R<=5.0:
                y.append(9)
        
        y=np.array(y)
        return feats,y,ratios
                
    def classify_to_speedup_bins(self,M):
        X,y,ratios=self.load_multiclass_data()
        X_train, X_test, Y_train, Y_test,R_train,R_test= train_test_split(X, y,ratios, test_size=0.2)
        model=self.loadmodel(M,"multiclass")
        model.fit(X_train,Y_train)
        preds=model.predict(X_test)
        accuracy=accuracy_score(preds,Y_test)
        return model,accuracy
    
    def multi_class_kfoldcv(self,n,M):
        X,y,ratios=self.load_multiclass_data()
        kf=KFold(n_splits=n,shuffle=True)
        tot_accuracy=0.0
        for train_index,test_index in kf.split(X):
            X_train,X_test=X[train_index],X[test_index]
            y_train,y_test=y[train_index],y[test_index]
            model=self.loadmodel(M,'classification')
            model.fit(X_train,y_train)
            preds=model.predict(X_test)
            tot_accuracy+=accuracy_score(preds,y_test)
        tot_accuracy/=n
        return tot_accuracy 
