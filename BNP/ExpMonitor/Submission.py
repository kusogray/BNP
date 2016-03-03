'''
Created on Jan 24, 2016

@author: yu1
'''
import os
from BNP.DataCollector.DataReader import DataReader as DataReader
from BNP.Curiosity.ExperimentPlayGround import exp
import pandas as pd
import csv
import os

_basePath =""
if os.name == 'nt':
    _basePath = "D:\\Kaggle\\Telstra\\"
else:
    _basePath = "/Users/whmou/Kaggle/Telstra/"
testPath = _basePath + "test6.csv"    # take id list only
testPath2 = _basePath + "test11.csv"   

samplePath = _basePath + "sample_submission.csv" 
outputPath = _basePath+"temp_submission4.csv"

if __name__ == '__main__':
    print "start to make submission version:", outputPath
    dr = DataReader()
    dr.readInCSV(testPath, "test")
    idList =  dr._testDataFrame[dr._testDataFrame.columns[0]]
    
    
    dr2= DataReader()
    dr2.readInCSV(testPath2, "test")
    
    
    dr3= DataReader()
    dr3.readInCSV(samplePath, "test")
    sampleIdList =  dr3._testDataFrame[dr3._testDataFrame.columns[0]]
    
    tmp = pd.DataFrame(exp())
    ansArr = pd.concat([idList, tmp], axis=1)
    print ansArr
    
    outputAnsArr = ansArr
    
    
    file = open(outputPath, 'w')
   
    
    
#     ansDf = ansArr
#     for i1 in range(0, len(sampleIdList)):
#         tmpSampleId = sampleIdList[i1]
#         for i2 in range (0,len(sampleIdList) ):
#             
#             tmpId = ansDf[ansDf.columns[0]][i2]
#             if tmpSampleId == tmpId:
#                 tmpOut = str(tmpId) + "," + str(ansDf[ansDf.columns[1]][i2]) + "," \
#                 + str(ansDf[ansDf.columns[2]][i2]) + "," +  str(ansDf[ansDf.columns[3]][i2]) + "\n"
#                 file.write(tmpOut)
#                 break
#         print i1
#     file.close()
    
    os.startfile('D:\\123.m4a')
    