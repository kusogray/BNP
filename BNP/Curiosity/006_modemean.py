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


    
from BNP.util.CustomLogger import info as log
from BNP.DataCollector.DataStratifier import stratifyData
from BNP.Resources import Config
from BNP.DataCollector.DataReader import DataReader as DataReader
from BNP.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
from BNP.util.CustomLogger import musicAlarm
from BNP.util.ModelUtils import loadModel
from BNP.util.ModelUtils import getMatchNameModelPath
from BNP.util.ModelUtils import deleteModelFiles
import pandas as pd
from BNP.Bartender.Blender import Blender
import random
import numpy as np
import xgboost as xgb


if __name__ == '__main__':
    
    
    log('Load data...')
    expInfo = "006_modemean"
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    trainPath = _basePath + "train.csv"
    testPath = _basePath + "test.csv"
    outputPath = _basePath + "ans.csv"
    
    dr = DataReader()
    dr.readInCSV(trainPath, "train")
    dr.readInCSV(testPath, "test")
    #dr.doSample(0.2)
    
    
    train = dr._trainDataFrame
    target = dr._ansDataFrame
    test = dr._testDataFrame
    id_test = dr._testIdDf
    numTrainDataRows = len(train)
    
    mergeDf = train.append(test)
    
    
    
    log('Fill With Mode or Median...')
    
    for i in range(0, len(mergeDf.columns)):
        tmpColName = mergeDf.columns[i]
        if str(mergeDf[mergeDf.columns[i]].dtype) =="float64":
            mergeDf[tmpColName] = mergeDf[tmpColName].fillna(mergeDf[tmpColName].median())    
        elif str(mergeDf[tmpColName].dtype) =="object":
            mergeDf[tmpColName] = mergeDf[tmpColName].fillna(train[tmpColName].mode()[0])
        else:
            mergeDf[tmpColName] = mergeDf[tmpColName].fillna(mergeDf[tmpColName].mode())
    
    log("Changing format...")
    
    for i in range(0, len(mergeDf.columns)):
        if str(mergeDf[mergeDf.columns[i]].dtype) =="float64":
            mergeDf[mergeDf.columns[i]]  = mergeDf[mergeDf.columns[i]].astype("float32")  
        elif str(mergeDf[mergeDf.columns[i]].dtype) =="int32":
            mergeDf[mergeDf.columns[i]]  = mergeDf[mergeDf.columns[i]].astype("int16")  
        elif str(mergeDf[mergeDf.columns[i]].dtype) =="object":
            mergeDf[mergeDf.columns[i]]  = pd.factorize(mergeDf[mergeDf.columns[i]])[0]
            mergeDf[mergeDf.columns[i]]  = mergeDf[mergeDf.columns[i]].astype("float32")  
    
    
    log("Split...")
            
    train = mergeDf.iloc[0:numTrainDataRows]    
    test =  mergeDf.iloc[numTrainDataRows:]  
    
    
    X = train
    Y = target
    
    
    log("start training...")
    
    
    fab = ModelFactory()
    fab._gridSearchFlag = True
    fab._singleModelMail = True
    fab._subFolderName = expInfo
    fab._n_iter_search = 500
    fab._expInfo = expInfo
    clf = fab.getXgboostClf(X, Y)
    
    
    train.to_csv(_basePath +"full_train.csv")
    test.to_csv(_basePath +"full_test.csv")
    
    log('Predict...')
    y_pred = []
    ans = pd.DataFrame(clf.predict(xgb.DMatrix(test)))
    
    y_pred = clf.predict(xgb.DMatrix(test))
        
    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred}).to_csv(outputPath,index=False)
    musicAlarm()
        