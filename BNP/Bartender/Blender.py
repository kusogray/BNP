'''
Created on Feb 7, 2016

@author: whmou
'''

from BNP.util.CustomLogger import info as log
from BNP.util.CustomLogger import mail
from BNP.Resources import Config
from BNP.DataCollector.DataReader import DataReader as DataReader
from BNP.util.CustomLogger import musicAlarm
from scipy.stats import randint as sp_randint
import numpy as np
import random
import pandas as pd
import math
import time
import sys

class Blender(object):
    '''
    classdocs
    '''
    _clfNameList, _predictDfList, _ansDf = [], [], []
    _currentMin = sys.maxint
    _lowProbIdList = []
    _bestParamList =[]

    def __init__(self, clfNameList, inputPredictDfList, inputAnsDf):
        '''
        Constructor
        '''
        self._predictDfList = inputPredictDfList
        self._ansDf = inputAnsDf
        self._clfNameList = clfNameList
        self._lowProbThreshold = 0.3
    
    
    def autoFlow (self, numIter, outputPath):
        log("Start blending autoFlow, num of Iter: " , numIter)
        start = time.time()
        distinctModels = len(self._clfNameList)
        tmpResultList =[]
        tmpRandomWeightList =[]
        tmpBlendedDfList =[]
        for i in range (0, numIter):
            tmpWeightList = self.getRandomWeightList(distinctModels)
            tmpRandomWeightList.append(tmpWeightList)
            tmpDf = self.doBlending(tmpWeightList)
            tmpBlendedDfList.append(tmpDf)
            tmpResultList.append(self.calLogLoss(tmpDf))
    
        
        idList = np.array(tmpResultList).argsort()[:3]
        firstFlag = True
        finalDf = []
        logResult =[]
        for id in idList:
            if firstFlag == True:
                finalDf = tmpBlendedDfList[id]
                self._bestParamList = tmpRandomWeightList[id]
                firstFlag = False
            log ("logloss: " , tmpResultList[id] , "blender param: " , tmpRandomWeightList[id])
            logResult.append ( (tmpResultList[id] , tmpRandomWeightList[id]))
        mail("Blender Top3: " ,logResult,  self._clfNameList)
        log("clfNameList = ", self._clfNameList)
        log ("low prob. id list (in 1st): #", len(self._lowProbIdList) , ", ", self._lowProbIdList)
        log("End blending autoFlow, num of Iter: " , numIter, " cost: ", time.time() - start , " sec") 
        
        finalDf.to_csv(outputPath, sep=',', encoding='utf-8')  
           
    def doBlending(self, inputWeightList):
        
        rtnDf = []
        tmpList = self._predictDfList[:]
        # print self._predictDfList
        for i, tmpDf in enumerate (self._predictDfList):
            tmpList[i] = tmpDf.multiply(inputWeightList[i])
            
                
        for i, tmpDf in enumerate (tmpList):
            if i == 0:
                rtnDf = tmpList[i]
            else:
                rtnDf = rtnDf.add(tmpList[i])
        
        
                    
        return rtnDf
    
    def getRandomWeightList(self, number):
        rtnList = []
        for i in range(0, number):
            rtnList.append(random.randint(1, random.randint(2,500)))
        tmpSum = np.sum(rtnList)
        rtnList = np.array(rtnList)
        rtnList = 1.0 / np.sum(rtnList, axis=0) * rtnList  
        return rtnList
        
    def calLogLoss(self, inputDf):
        N = len(self._ansDf)
        
        tmp = -1.0 / float(N)
        
        tmpSum = 0.0
        tmpLowProbIdList = []
        for i in range(0, N):
            tmpAns = self._ansDf[i]
            #print inputDf[inputDf.columns[tmpAns]][i]
            #print math.log(inputDf[inputDf.columns[tmpAns]][i])
            tmpVal = inputDf[inputDf.columns[tmpAns]][i]
            if tmpVal < self._lowProbThreshold:
                tmpLowProbIdList.append(i)
            if tmpVal >0:
                tmpSum += math.log(inputDf[inputDf.columns[tmpAns]][i])
                
        if tmpSum < self._currentMin:
            self._currentMin = tmpSum
            self._lowProbIdList = tmpLowProbIdList
            
        return  tmpSum* tmp     
        
if __name__ == '__main__':
    
    # 1. read data
#     dr = DataReader()
    _basePath = Config.FolderBasePath + "008_blender" + Config.osSep
#     path = _basePath + "blendTest.csv"
#     outputPath = _basePath + "blendTestOut.csv"
#     dr.readInCSV(path, "train")
#     newX, newY = dr._trainDataFrame, dr._ansDataFrame
#     
#     
#     dr2 = DataReader()
#     dr2.readInCSV(path, "train")
#     newX2, newY2 = dr2._trainDataFrame, dr2._ansDataFrame
#     
#     predictDfList = []
#     predictDfList.append(newX)
#     predictDfList.append(newX2)
    clfNameList = []
    clfNameList.append("test")
    clfNameList.append("test2")
#     
#     b1 = Blender(clfNameList, predictDfList, newY)
#     inputWeightList = b1.getRandomWeightList(2)
#     #print inputWeightList
#     tmpDf = b1.doBlending(inputWeightList)
#     #b1.calLogLoss(tmpDf)
#     b1.autoFlow(11, outputPath)
    
    dr3 = DataReader()
    path = _basePath + "testLogLoss.csv"
    dr3.readInCSV(path, "train")
    predictDfList = []
    predictDfList.append(dr3._trainDataFrame)
    b2 = Blender(clfNameList, predictDfList, dr3._ansDataFrame)
    print b2.calLogLoss(dr3._trainDataFrame)
    
