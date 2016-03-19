'''
Created on Jan 31, 2016
author: whmou

Jan 31, 2016     1.0.0     Init.

'''

from sklearn.externals import joblib
from BNP.util.CustomLogger import info as log
from BNP.Resources import Config
import os
import time
import fnmatch
import math
import shutil

## note that remove search contains subfolders
def deleteModelFiles(findFolderPath):
    for dirpath, dirnames, filenames in os.walk(findFolderPath):
        for file in filenames:
            if fnmatch.fnmatch(file, '*.model'):
                os.remove(os.path.join(dirpath, file))
            if fnmatch.fnmatch(file, '*.npy'):
                os.remove(os.path.join(dirpath, file))
                
def getMatchNameModelPath(findFolderPath, likeName):
    rtnStr = ""
    if likeName.strip():
        likeName = "*" + likeName +"*.model"
    for file in os.listdir(findFolderPath):
        if fnmatch.fnmatch(file, likeName):
            rtnStr = file
            
    return rtnStr

def getDumpFilePath(clfName, expInfo, subFolderName):

        expInfoFolder = Config.FolderBasePath + expInfo + Config.osSep + "models" + Config.osSep
        if subFolderName.strip():
            expInfoFolder += subFolderName + Config.osSep
        if not os.path.exists(expInfoFolder):
            os.makedirs(expInfoFolder)
        return expInfoFolder + "(" + clfName + ')_(' + str(time.strftime("%Y-%m-%d")) + '_' + str((time.strftime("%H_%M_%S"))) +').model'


def dumpModel(clf, clfName, expInfo, subFolderName):
    
    tmpDumpPath = getDumpFilePath(clfName, expInfo, subFolderName)
    log("Start dump ",clfName, " to " + tmpDumpPath)
    log("Exp info: ",expInfo)
    joblib.dump(clf, tmpDumpPath)
    log("Dump ",clfName, " successfully")
    

def loadModel(modelPath):
    
    log("Start load model: ", modelPath)
    clf = joblib.load( modelPath )
    return clf


def moveFileToPath(srcFile, destPath):
    shutil.move(srcFile, destPath)

     
def calLogLoss(inputDf, _ansDf):
        N = len(_ansDf)
        
        tmp = -1.0 / float(N)
        
        tmpSum = 0.0
        _ansDf = _ansDf.tolist()
        for i in range(0, N):
            tmpAns = _ansDf[i]
            tmpVal = inputDf[inputDf.columns[tmpAns]][i]
            if tmpVal >0:
                tmpSum += math.log(tmpVal)

        return  tmpSum* tmp     

if __name__ == '__main__':
    #log(getDumpFilePath("test", "test", " "))
    #getMatchNameModelPath("/Users/whmou/Kaggle/Telstra/010_stack_each_feature/models/log_feature", "Xgboost")
    #deleteModelFiles("/Users/whmou/Kaggle/Telstra/011_remove_one_hot/models/")
    
    moveFileToPath("F:\\test.model", "D:\\")
            