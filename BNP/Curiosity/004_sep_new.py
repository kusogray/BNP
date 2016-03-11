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
    numTrainDataRows = len(train)
    
    
    log('Label v1~v9 non empty...')
    
    selectedRowList =[]
    selectedTestRowList =[]
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
    
    
    for i in range(0, len(test["v1"])):
        isNan1 = str(test.iloc[i]["v1"]) =="nan"
        isNan2 = str(test.iloc[i]["v2"]) =="nan"
        isNan4 = str(test.iloc[i]["v4"]) =="nan"
        isNan5 = str(test.iloc[i]["v5"]) =="nan"
        isNan6 = str(test.iloc[i]["v6"]) =="nan" 
        isNan7 = str(test.iloc[i]["v7"]) =="nan"
        isNan8 = str(test.iloc[i]["v8"]) =="nan"
        isNan9 = str(test.iloc[i]["v9"]) =="nan"
        
        
        if  isNan1 == False and isNan2 == False and isNan4 == False and isNan5 == False and \
            isNan6 == False and isNan7 == False and isNan8 == False and isNan9 == False : 
            selectedTestRowList.append(i)
            
    # Open a file
#     fo = open("F:\\selectedRow.txt", "w+")
#     
#     # Write sequence of lines at the end of the file.
#     for tmpStr in selectedRowList:
#         line = fo.writelines( str(tmpStr) )
#         fo.writelines( "\n" )
#     
#     # Close opend file
#     fo.close()

    
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
    
    train_1 = train.iloc[np.asarray(selectedRowList)]
    target_1 = target.iloc[np.asarray(selectedRowList)]         
    
    train_2 = train.drop(np.asarray(selectedRowList))
    target_2 = target.drop(np.asarray(selectedRowList))         
    
    X = train_1
    Y = target_1
    
    
    log("start training...")
    
    
    fab = ModelFactory()
    fab._gridSearchFlag = True
    fab._singleModelMail = True
    fab._subFolderName = expInfo
    fab._n_iter_search = 1
    fab._expInfo = expInfo
    clf_1 = fab.getXgboostClf(train_1, target_1)
    
    
    fab2 = ModelFactory()
    fab2._gridSearchFlag = True
    fab2._singleModelMail = True
    fab2._subFolderName = expInfo
    fab2._n_iter_search = 1
    fab2._expInfo = expInfo
    clf_2 = fab2.getXgboostClf(train_2, target_2)
    
    train_1.to_csv(_basePath +"train1.csv")
    train_2.to_csv(_basePath +"train2.csv")
    test.to_csv(_basePath +"full_test.csv")
    
    log('Predict...')
    y_pred = []
    ans_1 = pd.DataFrame(clf_1.predict(xgb.DMatrix(test)))
    ans_2 = pd.DataFrame(clf_2.predict(xgb.DMatrix(test)))
    
    for i in range(0, len(test["v1"])):
        if i in selectedTestRowList:
            y_pred.append(ans_1.iloc[i][0])
        else:
            y_pred.append(ans_2.iloc[i][0])
        
    #y_pred = pd.DataFrame(y_pred)
    log(y_pred)
    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred}).to_csv(outputPath,index=False)
    musicAlarm()
        