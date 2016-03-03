'''
Created on Jan 31, 2016
author: whmou

Jan 31, 2016     1.0.0     Init.

'''
import os

FolderBasePath =""
osSep ="/"
musicAlarmPath = ""
mailAccPath = ""
allTimeLogPath = ""
xgboostBestTmpCflPath =""


if os.name == 'nt':
    FolderBasePath = 'D:\\Kaggle\\BNP\\'
    osSep = "\\"
    musicAlarmPath = 'D:\\123.m4a'
    mailAccPath = "D:\\python_mail_setting.txt"
    allTimeLogPath = "D:\\Kaggle\\BNP\\log.txt"
    xgboostBestTmpCflPath = "F:\\xgboost_tmp_best.model"
    
else: 
    FolderBasePath ='/Users/whmou/Kaggle/BNP/'
    musicAlarmPath = '/Users/whmou/123.mp3'
    mailAccPath = "/Users/whmou/python_mail_setting.txt"
    allTimeLogPath = "/Users/whmou/Kaggle/BNP/log.txt"
    xgboostBestTmpCflPath = "/Users/whmou/Kaggle/BNP/xgboost_tmp_best.model"
