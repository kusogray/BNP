'''
Created on Jan 24, 2016

@author: whmou
'''
import pandas as pd
import numpy as np

def mergeData():
    path = "/Users/whmou/Kaggle/Telstra/event_type.csv"
    pathMain = "/Users/whmou/Kaggle/Telstra/train_merge.csv"
    tmpColumnPrefix = "event_type_"
    df = pd.read_csv(path, header=0, sep=',')
    dfMain = pd.read_csv(pathMain, header=0, sep=',')
    for i in range (1,54):
        tmpColName = tmpColumnPrefix + str(i)
        dfMain[tmpColName] = 0


    for i1 in range(0, len(dfMain[dfMain.columns[0]] )):
        for i2 in range(0, len(df[df.columns[0]] )):
            tmpMainId = dfMain[dfMain.columns[0]][i1]
            tmpId = df[df.columns[0]][i2]
            if  tmpMainId == tmpId:
                tmpVal = df[df.columns[1]][i2]
                tmpColName = tmpColumnPrefix + str(tmpVal)
                dfMain[tmpColName][i1] = 1
            if tmpMainId > tmpId:
                break

    print dfMain
    
    
if __name__ == '__main__':
    mergeData()