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
from BNP.util.CustomLogger import info as log
from BNP.DataCollector.DataStratifier import stratifyData
from BNP.Resources import Config
from BNP.DataCollector.DataReader import DataReader as DataReader
from BNP.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
from BNP.util.CustomLogger import musicAlarm
from BNP.util.ModelUtils import loadModel
from BNP.util.ModelUtils import getMatchNameModelPath
from BNP.util.ModelUtils import deleteModelFiles
from BNP.util.ModelUtils import calLogLoss
import pandas as pd
from sklearn import cross_validation
from BNP.Bartender.Blender import Blender

if __name__ == '__main__':
    
    log('Load data...')
    expInfo = "009_chk_extra_tree"
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    trainPath = _basePath + "train.csv"
    testPath = _basePath + "test.csv"
    outputPath = _basePath + "ans.csv"
    
    
    
    dr = DataReader()
    dr.readInCSV(trainPath, "train")
    dr.readInCSV(testPath, "test")
    
    
    testLen = 800
    train = dr._trainDataFrame
    target = dr._ansDataFrame
    test = dr._testDataFrame
    id_test = dr._testIdDf
    
#     train = dr._trainDataFrame[:testLen]
#     target = dr._ansDataFrame[:testLen]
#     test = dr._testDataFrame[:testLen]
#     id_test = dr._idDataFrame[:testLen]
#     
    numTrainDataRows = len(train)
    
    
    log('drop useless columns...')

    train = train.drop(['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
    test = test.drop(['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
    
    
    
    
    log('Fill NA...')
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
                train.loc[train_series.isnull(), train_name] = -999 
            #and Test
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = -999
                
                
    extc = ExtraTreesClassifier(n_estimators=850,max_features= 60,criterion= 'entropy',min_samples_split= 4,
                                 max_depth= 40, min_samples_leaf= 2, n_jobs = 3)    
#     
#     scores = cross_validation.cross_val_score(extc, train, target, cv=4)
#     log("scores: " , scores)
    
    
    #cutPoint = int(len(train)*0.75)
    X_train = train
    X_test = test
    #id_test = id_test[cutPoint:]
    
    #X_target = target[:cutPoint]
    #ans = target[cutPoint:]
    
    #pd.DataFrame(X_test).to_csv(_basePath + "X_test.csv",index=False)
    #pd.DataFrame(ans).to_csv(_basePath + "ans2.csv",index=False)
    
    log('Training...')
    log('X_Train len: ', len(X_train))
    

    extc.fit(X_train,target) 
    
    log('Predict...')
    y_pred = extc.predict_proba(X_test)
    #log(calLogLoss(pd.DataFrame(y_pred), ans))
    
    #log("X_test: " , len(X_test), " ans: ", len(ans))
    #new_score = extc.score(X_test, ans) 
    #log("new_score: " , new_score)

    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv(outputPath,index=False)
    musicAlarm()

        