'''
Created on Feb 1, 2016
author: whmou

Feb 1, 2016     1.0.0     Init.

'''

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
import scipy as sp
import numpy as np


if __name__ == '__main__':
    
    
   # 1. read in data
    expNo = "015"
    expInfo = expNo + "_over_sampling" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    featureList = ["location", "event_type", "resource_type" , "severity_type", "log_feature"]
    ans1List = []
    ans2List = []
#     ansPath = _basePath + "014_ans_array.csv"
#     drAns = DataReader()
#     drAns.readInCSV(ansPath, "train")
#     newY = drAns._ansDataFrame

    tmpPath = _basePath + "train_v1.csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "train")
    newX = dr._trainDataFrame
    newY = dr._ansDataFrame
    
#     t = pd.get_dummies(newY)
#     finalList = []
#     for tmp in range(0,len(t)):
#         tmpList =[]
#         for i in range(0, len(t.ix[tmp])):
#             tmpList.append( int( t.ix[tmp][i]))
#         finalList.append(tmpList)
#     print finalList
    #exit()
     
 
    #print len(newX)        
    # Get all best model from newX
    fab = ModelFactory()
    fab._setXgboostTheradToOne = False
    fab._gridSearchFlag = True
    fab._subFolderName = "ismail3"  
    fab._n_iter_search = 1
    fab._expInfo = expInfo
  
    #clf = fab.getXgboostClf(newX, newY)
    clf = fab.getRandomForestClf(newX, newY)
    #print fab.getLogloss(clf,newX,newY)


    
    def llfun(act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(act*sp.log(pred) )
        ll = sum(ll)
        ll = ll * -1.0/len(act)
        return ll
    
    
    #musicAlarm()
    # Test all data
    modelList = ["Xgboost","Random_Forest","Extra_Trees", "K_NN", "Logistic_Regression"]
#     featureList = ["event_type", "log_feature", "resource_type", "severity_type"]
    
    #for tmpFeature in featureList:
    modelFolder = _basePath + "models" + Config.osSep + "ismail2" + Config.osSep
#     for tmpModel in modelList:  
#         curModel = tmpModel
#          
#     testPath = _basePath + "test_v1.csv"
#     dr = DataReader()
#     newX = dr.cvtPathListToDfList(testPath, "test")
#     curModel = "Random_Forest"
#         
#     modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
#     tmpOutPath = _basePath + expNo +"_" + curModel + "_test_ismail.csv"
#     tmpClf = loadModel( modelPath)
#     log(tmpClf.predict_proba(newX))
#     #outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
#     outDf = pd.DataFrame(tmpClf.predict_proba(newX))
#     outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    
     
    
    #print llfun( finalList, tmpClf.predict_proba(newX))
    
#     musicAlarm()
#     log("004 Done")