#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 05:04:35 2022

@author: dinho
"""

import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xg


"""Realiza a busca pelos melhores parâmetros
    Seleciona o arquivo de modelo
    Elimina os outliers
    Faz uma amostra dos dados
    Busca o melhor conjunto dentre as possíveis combinações
    
"""
nominal_columns = ["os_version", "health",'network_status']
def worker_regressor_model(model):
    search = {"modelo":"",
              "regressor":""
        }

    
    if (glob.glob(model)):
        medcouple_row=ns.medcouples_df.loc[ns.medcouples_df ["model"] == model.split("/")[1]]
       
        
    # model = "95_percent_models_full/A0001.csv"
        dataframe = pd.read_csv(model, sep=";")
        #stats ["model"] = model.split("/")[1].split(".")[0]
        dataframe
        
        
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
            #dataframe.drop(["timestamp"], axis=1, inplace=True)
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
            X_train, X_test, y_train, y_test = train_test_split(dataframe, target, 
                                                                test_size = 0.2, shuffle=False 
                                                                )
            
            dataframe["consume_time"] = target 
            
            X_train["consume_time"]=y_train
            upper = medcouple_row.upper.iloc[0]
            lower = medcouple_row.lower.iloc[0]          
           
            X_train = X_train[X_train["consume_time"] <= upper]
            X_train = X_train[X_train["consume_time"] >= lower]
            
            #X_train =  X_train.sample(n=10000, random_state=1)  
            #print (">>>>>  "+str(len(X_train)))
            
            X_train = X_train.sample(n=1500, random_state=2)  
            #print (">>>>>  "+str(len(X_train)))
            y_train = X_train["consume_time"]
            X_train.drop("consume_time", axis=1, inplace=True)
            print ("--------->"+str (len(X_train)))
            del X_test
            del y_test
            del dataframe
            del target
            if not X_train.empty:
                X_train = X_train.fillna(0)
                
            
                # Number of trees to be used
                xgb_n_estimators = {50,100,150,250,300}
                
                # Maximum number of levels in tree
                xgb_max_depth = [5,6,7,8,9,10] 
                
                # Minimum number of instaces needed in each node
                xgb_min_child_weight = [int(x) for x in np.linspace(1, 10, 10)]
                
                # Tree construction algorithm used in XGBoost
                #xgb_tree_method = ['auto', 'exact', 'approx', 'hist']
                
                # Learning rate
                xgb_eta = [x for x in np.linspace(0.1, 0.6, 6)]
                
               
                
                # Learning objective used
                xgb_objective = ['reg:squarederror', 'reg:squaredlogerror']
                xgb_subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
                
                # Create the grid
                
                xgb_grid = {'n_estimators': xgb_n_estimators,
                            'max_depth': xgb_max_depth,
                            'min_child_weight': xgb_min_child_weight,
                            #'tree_method': xgb_tree_method,
                            'eta': xgb_eta,
                            #'gamma': xgb_gamma,
                            'objective': xgb_objective,
                            'subsample': xgb_subsample
                            }
                xgb_base = xg.XGBRegressor()

                # Create the random search Random Forest
                xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions = xgb_grid, 
                                                n_iter = 60, cv = 2, verbose = 2, 
                                                random_state = 1, n_jobs = 1,  scoring="neg_mean_absolute_percentage_error")
                
                # Fit the random search model
                xgb_random.fit(X_train, y_train)
                
                # Get the optimal parameters
                xgb_random.best_params_
            # mlr_reg = LinearRegression()
            # rf_reg = RandomForestRegressor(random_state=42)
            # #xgb_reg = xg.XGBRegressor(random_state=42)
            
            # # Put the models in a list to be used for Cross-Validation
            # models = [mlr_reg, rf_reg] #xgb_reg]
            
            # # Run the Cross-Validation comparison with the models used in this analysis
            # comp, maes, mses, r2s, accs = cv_comparison(models, X_train, y_train, 3)
            # return comp
            # Create the model to be tuned

            search ["modelo"] = model
            search["regressor"] = xgb_random.best_params_
            # View the best parameters from the random search
            
            return search
from multiprocessing import Pool, Manager
import psutil

if __name__ == "__main__":
    mgr = Manager()
    ns = mgr.Namespace()
    medcouples = pd.read_csv("lower_upper_boundering_medcouple_detection.csv", sep=";")
    ns.medcouples_df = medcouples
    
    models_list = glob.glob("new_model_view_devices_100_processes_samples_full/*")
    models_list = [models_list[0]]
    
    
    pool = Pool(psutil.cpu_count())
    regressor_list_dataframe = pool.map(worker_regressor_model, models_list)
    pool.close()
    pool.join()
    regressor_list_dataframe = [x for x in regressor_list_dataframe  if x is not None]
    regressor_list_dataframe = pd.DataFrame(regressor_list_dataframe)
    #regressor_list_dataframe.to_csv("xgbooster_all/top_gti_samples.csv_tunning_xgbooster.csv", sep=";")
    
