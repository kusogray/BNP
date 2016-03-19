'''
Created on Feb 1, 2016
author: whmou

Feb 1, 2016     1.0.0     Init.

'''

# from BNP.util.CustomLogger import info as log
# from BNP.DataCollector.DataStratifier import stratifyData
# from BNP.Resources import Config
# from BNP.DataCollector.DataReader import DataReader as DataReader
# from BNP.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
# from BNP.util.CustomLogger import musicAlarm
# from BNP.util.ModelUtils import loadModel
# import pandas as pd


    
import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from BNP.Resources import Config
from BNP.util.CustomLogger import musicAlarm
from BNP.util.CustomLogger import info as log
from BNP.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory

if __name__ == '__main__':
    
    log('Load data...')
    expInfo = "007_rf_find"
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    trainPath = _basePath + "train.csv"
    testPath = _basePath + "test.csv"
    outputPath = _basePath + "ans.csv"
    
    train = pd.read_csv(trainPath)
    target = train['target'].values
    train = train.drop(['ID','target'],axis=1)
    test = pd.read_csv(testPath)
    id_test = test['ID'].values
    test = test.drop(['ID'],axis=1)
    
#     train = train[0:20]
#     target = target[0:20]
    
    log('Clearing...')
    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
        if train_series.dtype == 'O':
            #for objects: factorize
            train[train_name], tmp_indexer = pd.factorize(train[train_name])
            test[test_name] = tmp_indexer.get_indexer(test[test_name])
            #but now we have -1 values (NaN)
        else:
            #for int or float: fill NaN
            tmp_len = len(train[train_series.isnull()])
            if tmp_len>0:
                #print "mean", train_series.mean()
                train.loc[train_series.isnull(), train_name] = -9999 #train_series.mean()
            #and Test
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = -9999 #train_series.mean()  #TODO
    
    log("start training...")
    
    X_train = train
    X_test = test
    
    
    fab = ModelFactory()
    fab._gridSearchFlag = False
    fab._singleModelMail = True
    fab._subFolderName = "rf_find"  
    fab._n_iter_search = 3
    fab._expInfo = expInfo
    clf = fab.getRandomForestClf(X_train, target, None)
    importances = clf.feature_importances_
    log("rf importance: " , importances)
#     extc = ExtraTreesClassifier(n_estimators=700,max_features= 50,criterion= 'entropy',min_samples_split= 5,
#                                 max_depth= 50, min_samples_leaf= 5)      
#     
#     extc.fit(X_train,target) 
    
    log('Predict...')
    y_pred = clf.predict_proba(X_test)
    #print y_pred
    
    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('extra_trees.csv',index=False)
    musicAlarm()
        