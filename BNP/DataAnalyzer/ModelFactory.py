'''
Created on Jan 24, 2016

@author: whmou
'''
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

from BNP.util.CustomLogger import info as log
from BNP.util.CustomLogger import mail
from BNP.util.CustomLogger import musicAlarm
from BNP.DataCollector.DataStratifier import stratifyData
from BNP.util.ModelUtils import *
from sklearn.datasets import load_digits
from BNP.Resources import Config
from BNP.DataCollector.DataReader import DataReader as DataReader

import numpy as np
import pandas as pd
from operator import itemgetter
from random import randint
import random
import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randf
from sklearn.externals import joblib
import os
import sys

#http://docs.scipy.org/doc/numpy/reference/routines.random.html

from sklearn.ensemble import RandomForestClassifier as rf
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class ModelFactory(object):
    '''
    classdocs
    '''

    _gridSearchFlag = False
    _n_iter_search = 1
    _expInfo = "ExpInfo"
    _subFolderName =""
    _setXgboostTheradToOne = False
    _onlyTreeBasedModels = False
    _singleModelMail = False
    
    _bestScoreDict = {}
    _bestLoglossDict = {}
    _bestClf = {}   # available only when self._gridSearchFlag is True
    _basicClf = {}  # when self._gridSearchFlag is True, basic = best  
    _mvpClf = []
    
    _lastRandomSearchBestParam =[]
    
    
    def __init__(self):
        '''
        Constructor
        '''
    
    
    def getAllModels(self, X, Y):
        
        log("GetAllModels start with iteration numbers: " , self._n_iter_search)
        start = time.time()
        
        self._basicClf["Xgboost"] = self.getXgboostClf(X, Y)
        self._basicClf["Random_Forest"] = self.getRandomForestClf(X, Y)
        self._basicClf["Extra_Trees"] = self.getExtraTressClf(X, Y)
        
        if not self._onlyTreeBasedModels:
            self._basicClf["K_NN"] = self.getKnnClf(X, Y)
            self._basicClf["Logistic_Regression"] = self.getLogisticRegressionClf(X, Y)
            self._basicClf["Naive_Bayes"] = self.getNaiveBayesClf(X, Y)
        
        
        log("GetAllModels cost: " , time.time() - start , " sec")
        log(sorted(self._bestScoreDict.items(), key=lambda x: x[1] , reverse=True))
        mail(self._expInfo, sorted(self._bestScoreDict.items(), key=lambda x: x[1] , reverse=True) )
        log(self._expInfo, sorted(self._bestScoreDict.items(), key=lambda x: x[1] , reverse=True) )
        bestScoreList = sorted(self._bestScoreDict.items(), key=lambda x: x[1] , reverse=True)
        log("MVP clf is : ", bestScoreList[0][0])
        self._mvpClf = self._bestClf[bestScoreList[0][0]]
        log("GetAllModels end with iteration numbers: " , self._n_iter_search)


    def getLogloss(self, clf, X, Y):
        inputDf = pd.DataFrame(clf.predict_proba(X))
        #print inputDf
        return calLogLoss(inputDf, Y)

    # Utility function to report best scores
    def report(self, grid_scores, clfName, bestLogLoss, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        bestParameters = {}
        mailContent = ""
        for i, score in enumerate(top_scores):
            
            log("Model with rank: {0}".format(i + 1))
            log("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            log("Parameters: {0}".format(score.parameters))
            
            mailContent += str("Model with rank: {0}".format(i + 1)  )
            mailContent += "\n"
            mailContent += str("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores))   )
            mailContent += "\n"
            mailContent += str("Parameters: {0}".format(score.parameters)  )
            mailContent += "\n"
                    
            if i == 0:
                self._bestScoreDict[clfName] = score.mean_validation_score
                mailContent += str("Best CV score: ") + str ( score.mean_validation_score )
                mailContent += "\n"
                
            log("")
        #log (clfName , " best logloss: ", bestLogLoss)
        if (self._singleModelMail == True):
            mail("Single Model Done: ", clfName , ", ", mailContent)
        return bestParameters
    
    
    
        
    def doRandomSearch(self, clfName, clf, param_dist, X, Y):
        start = time.time()
        multiCores = -1
        if  clfName == "Logistic_Regression": 
            multiCores = 1
        if self._setXgboostTheradToOne == True and clfName =="Xgboost":
            multiCores = 1
            
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=self._n_iter_search, n_jobs=multiCores, scoring='log_loss')
        
        
        random_search.fit(X, Y)
        log(clfName + " randomized search cost: " , time.time() - start , " sec")
        self._bestClf[clfName] = random_search.best_estimator_
        self._bestLoglossDict[clfName] = self.getLogloss(self._bestClf[clfName], X, Y)
        self.report(random_search.grid_scores_, clfName, self._bestLoglossDict[clfName])
        
        random_search.best_params_
        
        dumpModel(random_search.best_estimator_, clfName, self._expInfo, self._subFolderName)
        self._lastRandomSearchBestParam = random_search.best_params_
        
        return random_search.best_estimator_
    
    # # 1. Random Forest
    def getRandomForestClf(self, X, Y, param_list):
        clfName = "Random_Forest"
        ## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        clf = rf(n_estimators=300, max_depth=None, min_samples_split=1, random_state=0, bootstrap=True, oob_score = True)
            
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            tmpLowDepth = 10
            tmpHighDepth = 50
            
            
            param_dist = {
                          "max_depth": sp_randint(tmpLowDepth, tmpHighDepth),
                          "max_features": sp_randf(0,1),
                          "min_samples_split": sp_randint(1, 11),
                          "min_samples_leaf": sp_randint(1, 11),
                          "criterion": ["gini", "entropy"], 
                          "n_estimators" : sp_randint(100, 300),
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
        
        else:    

            if param_list != -1:
                clf = ExtraTreesClassifier(param_list)
                clf.set_params(**param_list)
            clf.fit(X,Y)    
            
        return clf
    
    
    # # 2.xgboost
    def getXgboostClf(self, X, Y):
        clfName = "Xgboost"
        
        ## https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        tmpLowDepth = 10
        tmpHighDepth = 50
        
        num_class = len(set(Y))
        objective =""
        if len(set(Y)) <=2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        
        num_round = 120
        param = {'bst:max_depth':74, 
                 'bst:eta':0.05, 
                 'silent':1, 
                 'min_child_weight':2, 
                 'subsample': 0.6031536958709969,
                 #'colsample_bytree': 0.7,
                  'max_delta_step':9,
                   'gamma' : 3,
                   'eta' : 0.23833373077656667,
                    'eval_metric':'mlogloss',
                     'num_class':num_class ,
                      'objective':objective,
                      'alpha': 1,
                      'lambda': 1 }
        param['nthread'] = 4
        plst = param.items()
        
        clf = None
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            clf = self.doXgboostRandomSearch(X, Y, num_round)

        else:
            dtrain = xgb.DMatrix(X , label=Y)
            clf = xgb.train( plst, dtrain, num_round)
        #joblib.dump(clf, xgbModelPath)    
        return clf
    
    def getBestXgboostEvalScore(self, inputScoreList):
        minScore = sys.float_info.max
        minId =0
        for i, tmpScore in enumerate(inputScoreList):
            rndScore = float((tmpScore.split("\t")[1]).split(":")[1])
            if rndScore < minScore:
                minId = i+1
                minScore = rndScore
        return minId, minScore
    
            
    def doXgboostRandomSearch(self, X, Y, num_round):
        
        paramList = []
        bestScore = sys.float_info.max
        bestClf = None
        best_num_round=0 
        
        num_class = len(set(Y))
        objective =""
        if len(set(Y)) <=2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        
        for i in range(0, self._n_iter_search):
            log("xgboost start random search : " + str(i+1) + "/"+ str(self._n_iter_search))
            param = {}
            param['nthread'] = 4
            
            param['eta'] = random.uniform(0.15, 0.45)
            param['gamma'] = randint(0,3)
            param['max_depth'] = randint(8,20)
            param['min_child_weight'] = randint(1,3)
            param['eval_metric'] = 'mlogloss'
            param['max_delta_step'] = randint(1,10)
            param['objective'] = objective
            param['subsample'] = random.uniform(0.35, 0.75)
            param['num_class'] = num_class 
            param['silent'] = 1
            param['alpha'] = 1
            param['lambda'] = 1
            #param['early_stopping_rounds']=2
            plst = param.items()
        
            
            ori_X = X
            ori_Y = Y
            #dtest  = xgb.DMatrix( X.ix[sampleRows], label=sampleAnsDf)
            #dtrain  =  xgb.DMatrix( X.drop(sampleRows), label=Y.drop(sampleRows))
            #evallist  = [(dtest,'eval'), (dtrain,'train')]
            
            dtrain  =  xgb.DMatrix( X, label=Y)
            
            xgbCvResult =  xgb.cv(plst, dtrain, num_boost_round= num_round,  nfold=3)
            scoreList = xgbCvResult[xgbCvResult.columns[0]].tolist()
            new_num_round = scoreList.index(min(scoreList)) + 1 
            minScore = scoreList[new_num_round-1]
            
            tmpScore = minScore
            if  tmpScore < bestScore:
                #tmpSelfScore = calLogLoss(pd.DataFrame(bst.predict(dtest)), sampleAnsDf)
                #print "self best score:" + str(tmpSelfScore)
                log("xgb best score:" + str(minScore))
                log("xgb best num_round: " + str(new_num_round))
                log("xgb best param: " + str(plst))
                newDtrain = xgb.DMatrix(ori_X, label=ori_Y)
                bst = xgb.train(plst, newDtrain, new_num_round)
                
                bestScore = tmpScore
                bestClf = bst
                paramList = plst
                best_num_round = new_num_round
                joblib.dump(bst, Config.xgboostBestTmpCflPath)
                
                
        
        self.genXgboostRpt(bestClf, bestScore, paramList, best_num_round)
        return bestClf
        
    def genXgboostRpt(self, bestClf, bestScore, paramList, best_num_round):
        dumpModel(bestClf, "Xgboost", self._expInfo, self._subFolderName)
        log("Native Xgboost best score : ", bestScore, ", param list: ", paramList, "best_num_round: ", best_num_round)
        if self._singleModelMail == True:
            mail("Xgboost Done" ,"Native Xgboost best score : " + str( bestScore) + ", param list: " + str( paramList) + "best_num_round: ", best_num_round)
        
    # # 3.Extra Trees
    def getExtraTressClf(self, X, Y, param_list=-1):
        clfName = "Extra_Trees"
        
        ## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        clf = ExtraTreesClassifier(
                                    n_estimators=10, 
                                    criterion='gini', 
                                    max_depth=None, 
                                    min_samples_split=2, 
                                    min_samples_leaf=1, 
                                    min_weight_fraction_leaf=0.0, 
                                    max_features='auto', 
                                    max_leaf_nodes=None, 
                                    bootstrap=False, 
                                    oob_score=False, 
                                    n_jobs=1, 
                                    random_state=None, 
                                    verbose=0, 
                                    warm_start=False, 
                                    class_weight=None)
        
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            tmpLowDepth = int(len(X.columns) * 0.7)
            tmpHighDepth = int(len(X.columns) )
            
            param_dist = {
                          "max_depth": sp_randint(tmpLowDepth, tmpHighDepth),
                          "max_features": sp_randf(0,1),
                          "min_samples_split": sp_randint(1, 11),
                          "min_samples_leaf": sp_randint(1, 11),
                          "bootstrap": [True, True],
                          "criterion": ["gini", "entropy"], 
                          "oob_score":[True, True],
                          "n_estimators" : sp_randint(800, 1200),
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
        else:    

            if param_list != -1:
                clf = ExtraTreesClassifier(param_list)
                clf.set_params(**param_list)
            clf.fit(X,Y)
        
        return clf
    
    
    # # 4.KNN
    def getKnnClf(self, X, Y):
        clfName = "K_NN"
        
        ## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        clf = KNeighborsClassifier(
                                n_neighbors=5, 
                                weights='uniform', 
                                algorithm='auto', 
                                leaf_size=30, 
                                p=2, 
                                metric='minkowski', 
                                metric_params=None, 

                                )
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            
            param_dist = {
                          "n_neighbors": sp_randint(4, 8),
                          "weights": ['uniform', 'uniform'],
                          "leaf_size": sp_randint(30, 60),
                          "algorithm": ['auto', 'auto'],
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
            
        return clf
    
    # # 5.Logistic Regression
    def getLogisticRegressionClf(self, X, Y):
        clfName = "Logistic_Regression"
        
        ## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        clf = LogisticRegression(
                                penalty='l2', 
                                dual=False, 
                                tol=0.0001, 
                                C=1.0, 
                                fit_intercept=True, 
                                intercept_scaling=1, 
                                class_weight=None, 
                                random_state=None, 
                                solver='liblinear', 
                                max_iter=100, 
                                multi_class='ovr', 
                                verbose=0, 


                                )
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            
            param_dist = {
                          "penalty": ['l2', 'l2'],
                          "C": sp_randf(1.0,3.0),
                          "solver": [ 'lbfgs', 'liblinear'],
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
        else:
            clf.fit(X,Y)
                
        return clf
    
    # # 6. Naive Bayes
    def getNaiveBayesClf(self, X, Y):
        clfName = "Naive_Bayes"
        
        ## http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
        clf = GaussianNB()
        clf = clf.fit(X, Y)
        scores = cross_val_score(clf, X,  Y )
        log( clfName + " Cross Validation Precision: ", scores.mean() )
        self._bestScoreDict[clfName] = scores.mean()
            
        return clf
    
if __name__ == '__main__':
#     fab = ModelFactory()
#     fab._gridSearchFlag = True
    
    #log (sp_randf)
    

    
    digits = load_digits()
    X = digits.data
    Y = digits.target
    #X, Y =  pd.DataFrame([9,9,9,9,8,8,8,7,7,7,6,6]), pd.DataFrame([0,0,0,0,1,1,1,2,2,2,3,3])
    #X, Y =  pd.DataFrame([1,2,3,4,5,6,7,8,9,10,11,12]), pd.DataFrame([1,2,3,4,5,6,7,8,9,10,11,12])
    #newX, newY =  stratifyData(X,Y, 0.4)
#     clf = fab.getNaiveBayesClf(X, Y)
#     clf2 = fab.getKnnClf(X, Y)
    #clf3 = fab.getRandomForestClf(X, Y)
#     x= clf.predict_proba(X)
#     log( x)
    #log(fab._bestScoreDict)
#     #log(fab._bestClf)
#     log( fab._bestClf['Random Forest'].predict_proba(X))
    #newX, newY = stratifyData(X, Y, 0.4)
    newX, newY = X, Y
    #print newX
    fab = ModelFactory()
    fab._gridSearchFlag = True
    fab._n_iter_search = 1
    fab._expInfo = "001_location_only" 
    print newX
    #print newY
    fab.getAllModels(newX, newY)
    #fab.getRandomForestClf(newX, newY)

    bestClf = fab._mvpClf
    log(bestClf.predict_proba(newX))
    #log(sorted(fab._bestScoreDict.items(), key=lambda x: x[1] , reverse=True) )
    #log(fab._bestClf['Random Forest'].predict_proba(X))
    #dumpModel(clf3, "Random_Forest", "ExpTest")
    #log("haha")
    #log(getDumpFilePath( "Random_Forest", "haha Tets"))      
    #musicAlarm()
