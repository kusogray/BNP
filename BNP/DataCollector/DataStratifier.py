'''
Created on Feb 1, 2016
author: whmou

Feb 1, 2016     1.0.0     Init.

'''

from BNP.util.CustomLogger import info as log
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
import random
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np


def stratifyData(X, Y, n): 
    sss = StratifiedShuffleSplit(Y, 1, test_size=n, random_state=random.randint(1, 9999))
    for trainIdx, testIdx  in sss:
        newX = trainIdx
        #newY = testIdx
    
    return X.iloc[newX] , Y.iloc[newX]
    
if __name__ == '__main__':
    digits = load_digits()
    X, Y =  pd.DataFrame([9,9,9,9,8,8,8,7,7,7,6,6]), pd.DataFrame([0,0,0,0,1,1,1,2,2,2,3,3])
    newX, newY =  stratifyData(X,Y, 0.4)
    print newX
    print newY