#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 05:04:35 2022

@author: dinho
"""

import pandas as pd
import glob
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV



nominal_columns = ["os_version", "health",'network_status']

"""Realiza a busca pelos melhores parâmetros
    Seleciona o arquivo de modelo
    Elimina os outliers
    Faz uma amostra dos dados
    Busca o melhor conjunto dentre as possíveis combinações
    
"""
def worker_regressor_model(model):
    search = {"modelo":"",
              "regressor":""
        }
    
    if (glob.glob(model)):
        medcouple_row=ns.medcouples_df.loc[ns.medcouples_df ["model"] == model.split("/")[1]]
       
        
    # model = "95_percent_models_full/A0001.csv"
        dataframe = pd.read_csv(model, sep=";")
        #stats ["model"] = model.split("/")[1].split(".")[0]
        
        
        
        if len(dataframe.index) > (len(dataframe.columns)-1)*10:
            dataframe=dataframe.sort_values(by=['timestamp'])
            #stats["samples eliminados por 1 minuto"]=len(dataframe[dataframe["consume_time"] <= 60])
            
            for nominal_column in nominal_columns:
                if nominal_column in dataframe.columns:
                    dataframe[nominal_column] = dataframe [nominal_column].astype(str)
                    dataframe[nominal_column]  = dataframe[nominal_column].str.replace(" ","")
                    nominal_dataframe=pd.get_dummies(dataframe[nominal_column], prefix=nominal_column)
                    dataframe.drop(nominal_column, axis=1, inplace=True)
                    dataframe = pd.concat([dataframe,nominal_dataframe], axis=1)
            
            dataframe.drop(list(dataframe.filter(regex = 'Unna')), axis = 1, inplace=True)
            
            if "brand" in dataframe.columns:
                dataframe.drop(["brand"], axis=1, inplace=True)
            if "manufacturer" in dataframe.columns:
                dataframe.drop(["manufacturer"], axis=1, inplace=True)
            if "model" in dataframe.columns:
                dataframe.drop(["model"], axis=1, inplace=True)
            if "timezone" in dataframe.columns:
                dataframe.drop(["timezone"], axis=1, inplace=True)
            if "country_code" in dataframe.columns:
                dataframe.drop(["country_code"], axis=1, inplace=True)

            dataframe=dataframe.sort_values(by=['timestamp'])
            dataframe.drop(["timestamp"], axis=1, inplace=True)
            target = dataframe["consume_time"]
            dataframe=dataframe.drop("consume_time", axis=1)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(dataframe, target, test_size=0.2, random_state=1)
            dataframe["consume_time"] = target 
            
            X_train["consume_time"]=y_train
            upper = medcouple_row.upper.iloc[0]
            lower = medcouple_row.lower.iloc[0]          
           
            X_train = X_train[X_train["consume_time"] <= upper]
            X_train = X_train[X_train["consume_time"] >= lower]
            
            #X_train =  X_train.sample(n=10000, random_state=1)  
            #print (">>>>>  "+str(len(X_train)))
            
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
            
 
            #print (">>>>>  "+str(len(X_train)))

            y_train = X_train["consume_time"]
            X_train.drop("consume_time", axis=1, inplace=True)
            
            del X_test
            del y_test
            del dataframe
            del target
            if not X_train.empty:
                X_train = X_train.fillna(0)
                
            
            # Number of trees in Random Forest
            # verificar se são necessários tantos
# Hyper parameters range intialization for tuning 

                dt_grid={"splitter":["best","random"],
                           "max_depth" : [5,6,7,8,9,10],
                           "min_samples_split":[int(x) for x in np.linspace(2, 10, 9)],
                           "min_samples_leaf": [int(x) for x in np.linspace(2, 10, 9)],
                           "min_impurity_decrease":[0.0, 0.05, 0.1],
                           "criterion":["mse", "mae"]}
            # mlr_reg = LinearRegression()
            # rf_reg = RandomForestRegressor(random_state=42)
            # #xgb_reg = xg.XGBRegressor(random_state=42)
            
            # # Put the models in a list to be used for Cross-Validation
            # models = [mlr_reg, rf_reg] #xgb_reg]
            
            # # Run the Cross-Validation comparison with the models used in this analysis
            # comp, maes, mses, r2s, accs = cv_comparison(models, X_train, y_train, 3)
            # return comp
            # Create the model to be tuned
            dt_base = DecisionTreeRegressor()
            
            # Create the random search Random Forest
            dt_random = RandomizedSearchCV(estimator = dt_base, param_distributions = dt_grid, 
                                           n_iter = 60, cv = 2, verbose = 1, random_state = 1, 
                                           n_jobs = 3, scoring="neg_mean_absolute_percentage_error")
            
            # Fit the random search model
            dt_random.fit(X_train, y_train)
            search ["modelo"] = model
            search["regressor"] = dt_random.best_params_
            # View the best parameters from the random search
            
            return search
from multiprocessing import Pool, Manager
import psutil

if __name__ == "__main__":
    mgr = Manager()
    ns = mgr.Namespace()
    medcouples = pd.read_csv("lower_upper_boundering_medcouple_detection.csv", sep=";")
    ns.medcouples_df = medcouples
    
    
    # decision_tree_df = pd.read_csv("decision_tree_all/decision_tree_model_pred_df.csv", sep=";")
    # decision_tree_df = decision_tree_df.loc[decision_tree_df['tempo limite inferior'] != decision_tree_df['tempo limite superior']]
    # decision_tree_df = decision_tree_df[["model","samples totais"]]
    # decision_tree_df = decision_tree_df.sort_values(by='samples totais', ascending=False)
    # decision_tree_df=decision_tree_df.head(100)
    
    
    # models_list = list(decision_tree_df.model)
    # pool = Pool(psutil.cpu_count())
    # regressor_list_dataframe = pool.map(worker_regressor_model, models_list)
    # pool.close()
    # pool.join()
    # regressor_list_dataframe = [x for x in regressor_list_dataframe  if x is not None]
    # regressor_list_dataframe = pd.DataFrame(regressor_list_dataframe)
    # regressor_list_dataframe.to_csv("decision_tree_all/top_100_most_popular/top_100_most_popular_tunning_decision_tree.csv", sep=";")
    
    
    # decision_tree_df = pd.read_csv("decision_tree_all/decision_tree_model_pred_df.csv", sep=";")
    # decision_tree_df = decision_tree_df.loc[decision_tree_df['tempo limite inferior'] != decision_tree_df['tempo limite superior']]
    # decision_tree_df = decision_tree_df[["model","mean_absolute_percentage_error"]]
    # decision_tree_df = decision_tree_df.sort_values(by='mean_absolute_percentage_error', ascending=False)
    # decision_tree_df=decision_tree_df.head(100)
    
    
    models_list = glob.glob("new_model_view_devices_100_processes_samples_full/*")
    pool = Pool(psutil.cpu_count())
    regressor_list_dataframe = pool.map(worker_regressor_model, models_list)
    pool.close()
    pool.join()
    regressor_list_dataframe = [x for x in regressor_list_dataframe  if x is not None]
    regressor_list_dataframe = pd.DataFrame(regressor_list_dataframe)
    regressor_list_dataframe.to_csv("decision_tree_all/new_top_100_tunning_decision_tree.csv", sep=";")