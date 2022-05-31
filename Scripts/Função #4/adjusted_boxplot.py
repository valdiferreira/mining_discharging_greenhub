#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 13:10:45 2021

@author: dinho
"""
import numpy as np
import math

import weightedstats as ws

import glob
import pandas as pd
# from statsmodels.stats.stattools import medcouple

#https://github.com/tks1998/statistical-function-and-algorithm-ML-/blob/master/medcople.py

class Med_couple:
    
    def __init__(self,data):
        self.data = np.sort(data,axis = None)[::-1] # sorted decreasing  
        self.med = np.median(self.data)
        self.scale = 2*np.amax(np.absolute(self.data))
        self.Zplus = [(x-self.med)/self.scale for x in self.data if x>=self.med]
        self.Zminus = [(x-self.med)/self.scale for x in self.data if x<=self.med]
        self.p = len(self.Zplus)
        self.q = len(self.Zminus)
    
    def H(self,i,j):
        a = self.Zplus[i]
        b = self.Zminus[j]

        if a==b:
            return np.sign(self.p - 1 - i - j)
        else:
            return (a+b)/(a-b)

    def greater_h(self,u):

        P = [0]*self.p

        j = 0

        for i in range(self.p-1,-1,-1):
            while j < self.q and self.H(i,j)>u:
                j+=1
            P[i]=j-1
        return P

    def less_h(self,u):

        Q = [0]*self.p

        j = self.q - 1

        for i in range(self.p):
            while j>=0 and self.H(i,j) < u:
                j=j-1
            Q[i]=j+1
        
        return Q
    #Kth pair algorithm (Johnson & Mizoguchi)
    def kth_pair_algorithm(self):
        L = [0]*self.p
        R = [self.q-1]*self.p

        Ltotal = 0

        Rtotal = self.p*self.q

        medcouple_index = math.floor(Rtotal / 2)

        while Rtotal - Ltotal > self.p:

            middle_idx = [i for i in range(self.p) if L[i]<=R[i]]
            row_medians = [self.H(i,math.floor((L[i]+R[i])/2)) for i in middle_idx]

            weight = [R[i]-L[i] + 1 for i in middle_idx]

            WM = ws.weighted_median(row_medians,weights = weight)
            
            P = self.greater_h(WM)

            Q = self.less_h(WM)

            Ptotal = np.sum(P)+len(P) 
            Qtotal = np.sum(Q)

            if medcouple_index <= Ptotal-1:
                R = P.copy()
                Rtotal = Ptotal
            else:
                if medcouple_index > Qtotal - 1:
                    L = Q.copy()
                    Ltotal = Qtotal
                else:
                    return WM
        remaining = np.array([])
       
        for i in range(self.p):
            for j in range(L[i],R[i]+1):
                remaining = np.append(remaining,self.H(i,j))

        find_index = medcouple_index-Ltotal

        k_minimum_element = remaining[np.argpartition(remaining,find_index)]
        
        # print(find_index,'tim trong mang ',sorted(remaining))
        return k_minimum_element[find_index]
       
    def naive_algorithm_testing(self):
        result = [self.H(i,j) for i in range(self.p) for j in range(self.q)]
        return np.median(result)

""" Calcula o limite superior e inferior de cada modelo para retirada de outiers"""
from sklearn.model_selection import train_test_split
def worker_quantiles(model):
    adjust_box_plot = {
    "model":"",
    "lower":"",
    "upper":"",
    "medcouple":""
    }

    dataframe = pd.read_csv(model, sep=";", usecols=["timestamp", "consume_time"])
    dataframe=dataframe.sort_values(by=['timestamp'])
    dataframe.drop(["timestamp"], axis=1, inplace=True)
    target = dataframe["consume_time"]
    dataframe=dataframe.drop("consume_time", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target, test_size=0.2, random_state=1)


        
    X_train["consume_time"]=y_train
    del X_test
    del y_test
    del dataframe
    del target
    medcouple=Med_couple(X_train["consume_time"]).kth_pair_algorithm()
    Q1 = np.percentile(X_train["consume_time"],25)
    Q3 = np.percentile(X_train["consume_time"],75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    if (medcouple >=0):
        lower = Q1 - outlier_step*math.exp(-3.5*medcouple)
        upper = Q3 + outlier_step*math.exp(4*medcouple)
    if (medcouple <0):
        lower = Q1 - outlier_step*math.exp(-4*medcouple)
        upper = Q3 + outlier_step*math.exp(3.5*medcouple)
    adjust_box_plot["model"]=model.split("/")[1]
    adjust_box_plot["medcouple"]=medcouple
    adjust_box_plot ["lower"] = lower
    adjust_box_plot["upper"] = upper
   
    return adjust_box_plot




if __name__ == "__main__":
    
    all_models = glob.glob("new_model_view_devices_100_processes_samples_full/*")
    from multiprocessing import Pool
    import psutil
    pool = Pool(psutil.cpu_count())
    models_list_dataframe = pool.map(worker_quantiles, all_models)
    pool.close()
    pool.join()
    models_list_dataframe=[x for x in models_list_dataframe if x is not None]
    
    
    models_list_dataframe = pd.DataFrame(models_list_dataframe)
    models_list_dataframe.to_csv("lower_upper_boundering_medcouple_detection.csv", sep=";")
    
    
    


















