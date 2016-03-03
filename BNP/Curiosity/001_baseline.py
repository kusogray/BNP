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
import pandas as pd

if __name__ == '__main__':
    
    # 1. read in data
    log ("hello BNP!")
    