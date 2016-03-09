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
from test._mock_backport import inplace
import random
import numpy as np
import pickle


if __name__ == '__main__':
    
    
    log('Load data...')
    expInfo = "004_sep_new"
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
    
    selectedRowList =[]
    for i in range(0, len(train["v1"])):
        isNan1 = str(train.iloc[i]["v1"]) =="nan"
        isNan2 = str(train.iloc[i]["v2"]) =="nan"
        isNan4 = str(train.iloc[i]["v4"]) =="nan"
        isNan5 = str(train.iloc[i]["v5"]) =="nan"
        isNan6 = str(train.iloc[i]["v6"]) =="nan"
        isNan7 = str(train.iloc[i]["v7"]) =="nan"
        isNan8 = str(train.iloc[i]["v8"]) =="nan"
        isNan9 = str(train.iloc[i]["v9"]) =="nan"
        
        
        if  isNan1 == False and isNan2 == False and isNan4 == False and isNan5 == False and \
            isNan6 == False and isNan7 == False and isNan8 == False and isNan9 == False : 
            selectedRowList.append(i)
    
    
    # Open a file
    fo = open("F:\\selectedRow.txt", "w+")
    
    # Write sequence of lines at the end of the file.
    for tmpStr in selectedRowList:
        line = fo.writelines( str(tmpStr) )
        fo.writelines( "\n" )
    
    # Close opend file
    fo.close()
    #exit()

    train = train.iloc[np.asarray(selectedRowList)]
    target = target.iloc[np.asarray(selectedRowList)]
    
    mergeDf = train.append(test)
    
    log('Pre-Processing...')
    
    numTrainDataRows = len(train)
    categoryThreshold = 50
    toDropList = []
    
    
    
    
    for i in range(0, len(mergeDf.columns)):
        tmpColName = mergeDf.columns[i]
        mergeDf[tmpColName] = mergeDf[tmpColName].fillna("-9999")
        uniqueCnt = len(pd.unique(mergeDf[tmpColName]))
        
        if uniqueCnt <= categoryThreshold:
            toDropList.append(tmpColName)
            uniqueArr = pd.unique(mergeDf[tmpColName]).tolist()
            tmpDf = pd.get_dummies(mergeDf[tmpColName], prefix= tmpColName + "_onehot")
            mergeDf = pd.concat([mergeDf, tmpDf], axis = 1)
    
        else:
            replaceStr = str(mergeDf.iloc[0][tmpColName]).replace(".","").replace("-","")
            if not replaceStr.isdigit():
                mergeDf[tmpColName]  = pd.factorize(mergeDf[tmpColName])[0]
                
    for tmpName in toDropList:
        mergeDf = mergeDf.drop(tmpName,axis=1)        
            
    train = mergeDf.iloc[0:numTrainDataRows]    
    test =  mergeDf.iloc[numTrainDataRows:]  
    
    X = train
    Y = target
    
    X.to_csv(_basePath+"full_train.csv")
    test.to_csv(_basePath + "full_test.csv")
    
    samplePercent = 0.3
    sampleRows = np.random.choice(X.index, len(X)*samplePercent) 
    train_sample  =  X.ix[sampleRows]
    ans_sample = Y.ix[sampleRows]
    log("sample len: ", len(train_sample[train_sample.columns[0]]))
    
    train_sample.to_csv(_basePath+"sample_train.csv")
    
    log("start training...")
    
    
    fab = ModelFactory()
    fab._gridSearchFlag = True
    fab._singleModelMail = True
    fab._subFolderName = expInfo
    fab._n_iter_search = 20
    fab._expInfo = expInfo
    clf = fab.getExtraTressClf(train_sample, ans_sample)
    bestParam= fab._lastRandomSearchBestParam
    
    
    fab = ModelFactory()
    fab._gridSearchFlag = False
    fab._singleModelMail = True
    clf = fab.getExtraTressClf(train, target, bestParam)
    
    
    
    log('Predict...')
    y_pred = clf.predict_proba(test)
    
    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv(outputPath,index=False)
    musicAlarm()
        