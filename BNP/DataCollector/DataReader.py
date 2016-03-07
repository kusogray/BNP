'''
Created on Jan 24, 2016

@author: whmou
'''

import pandas as pd
import numpy as np
from BNP.util.CustomLogger import info as log

class DataReader(object):
    '''
    classdocs
    '''
    _trainDataFrame, _testDataFrame, _ansDataFrame = [],[],[]
    
    _trainSampleDf, _trainRestDf = [], []
    _ansSampleDf, _ansRestDf = [], []
    
    _testIdDf = []

    def __init__(self, ):
        '''
        Constructor
        '''
    def readInCSV(self, path, mode):
        # # 1. read csv data in
        df = pd.read_csv(path, header=0, sep=',')
        log("loading csv: " + path)
        if mode.lower() == "train":
            self._ansDataFrame = df[df.columns[1]]
            self._trainDataFrame = df[df.columns[2:]]
        else:
            self._testIdDf = df[df.columns[0]]
            self._testDataFrame = df[df.columns[1:]]
    
    
    
    def doSample(self, samplePercent):
        
        X = self._trainDataFrame
        Y = self._ansDataFrame
        sampleRows = np.random.choice(X.index, len(X)*samplePercent) 
                
        
        self._trainSampleDf  =  X.ix[sampleRows]
        self._ansSampleDf = Y.ix[sampleRows]
        
        self._trainRestDf  =  X.drop(sampleRows)
        self._ansRestDf = Y.drop(sampleRows)
        