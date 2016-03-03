'''
Created on Feb 4, 2016
author: whmou

Feb 4, 2016     1.0.0     Init.

'''



import pandas as pd
import time
from BNP.DataCollector.DataReader import DataReader as DataReader
import os
from BNP.Resources import Config
from BNP.util.CustomLogger import musicAlarm



_basePath = Config.FolderBasePath + "003_one_hot_event_type" + Config.osSep

_typeName = "event_type"
_eventTypePath = _basePath + _typeName + ".csv"
_pathMain = _basePath + "003_test_asis.csv"
_outputPathName = _basePath + "003_test_tobe.csv"
_mode = "test"



def oneHot():
    dr = DataReader()
    dr.readInCSV(_pathMain, _mode)
    tmpColumnPrefix = _typeName + "_"
    df = pd.read_csv(_eventTypePath, header=0, sep=',')
    if _mode == "train":
        processDf = dr._trainDataFrame
    else:
        processDf = dr._testDataFrame
        
    for i in range (1, 55):
        tmpColName = tmpColumnPrefix + "one_hot_" + str(i)
        processDf[tmpColName] = 0
    
    tmpLastI2 = 0
    for i1 in range(0, len(processDf[processDf.columns[0]])):

        tmpFlag = False
        for i2 in range(tmpLastI2, len(df[df.columns[0]])):
            tmpMainId = processDf[processDf.columns[0]][i1]
            tmpId = df[df.columns[0]][i2]
            tmpVal = df[df.columns[1]][i2]
            # tmpVal2= df[df.columns[2]][i2]
            if  tmpMainId == tmpId:
                tmpFlag = True
                print tmpVal
                processDf[processDf.columns[tmpVal + 394]][i1] = 1
            if tmpFlag == True and tmpMainId != tmpId:
                tmpLastI2 = i2
                break
            print i1, i2
    # outDf = pd.concat([dr._ansDataFrame, processDf], axis=1)
    outDf = processDf
    outDf.to_csv(_outputPathName, sep=',', encoding='utf-8')  
    # print dr._ansDataFrame

    
    
        
if __name__ == '__main__':
    
    start = time.time()
    
    oneHot()

    elapsed = time.time() - start
    print "elapsed:", elapsed , "sec"
    musicAlarm()


