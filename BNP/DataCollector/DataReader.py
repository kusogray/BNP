'''
Created on Jan 24, 2016

@author: whmou
'''

import pandas as pd
from BNP.util.CustomLogger import info as log

class DataReader(object):
    '''
    classdocs
    '''
    _trainDataFrame, _testDataFrame, _ansDataFrame = [],[],[]
    
    _trainSampleDf, _trainEvalSampleDf = [], []
    _ansSampleDf, _ansEvalSampleDf = [], []

    def __init__(self, ):
        '''
        Constructor
        '''
    def readInCSV(self, path, mode):
        # # 1. read csv data in
        df = pd.read_csv(path, header=0, sep=',')
        log("loading csv: " + path)
        if mode.lower() == "train":
            self._ansDataFrame = df[df.columns[0]]
            self._trainDataFrame = df[df.columns[1:]]
        else:
            self._testDataFrame = df
    
    def cvtPathListToDfList(self, inputPath, mode):
        self.readInCSV(inputPath, mode)
        if mode.lower() == "train":
            return self._trainDataFrame, self._ansDataFrame
        else:
            return self._testDataFrame
        
        