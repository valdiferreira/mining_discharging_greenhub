#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 07:57:16 2021

@author: dinho
"""

import pandas as pd

import xgboost as xg
import numpy as np

import shap
import glob

import ast

"""Realiza as predições
    Seleciona o arquivo de modelo
    Elimina os outliers
    Faz uma amostra dos dados
    Atribui os melhores meta-parâmetros para as funções dos algoritmos
    Realiza as predições
    Gera os arquivos com os dados dos do modelo
    Gera os arquivos com os valores SHAP
"""


'''setup model'''
nominal_columns = ["os_version", "health",'network_status']
def worker_regressor_model(modelo):
    model = modelo["modelo"]
    if (glob.glob(model)):
        medcouple_row=ns.medcouples_df.loc[ns.medcouples_df ["model"] == model.split("/")[1]]
       
        stats = {"model":"",
            "test media":"",
              "test mediana":"",
              "test desvio":"",
              "test minimo":"",
              "test maximo":"",
              "mean absolute error":"",
              "r score":"",
              "root mean squared error":"",
              "explained_variance_score":"",
              "samples totais":"",
              #"samples eliminados por 1 minuto":"",
              "samples eliminados por limite inferior":"",
              "samples eliminados por limite superior":"",
              "tempo limite inferior":"",
              "tempo limite superior":"",
              "medcouple":"",
              "mean_absolute_percentage_error":""
        
        }
    # model = "95_percent_models_full/A0001.csv"
        dataframe = pd.read_csv(model, sep=";")
        stats ["model"] = model.split("/")[1].split(".")[0]
        
        
        
        if len(dataframe.index) > (len(dataframe.columns)-1)*10:
            dataframe=dataframe.sort_values(by=['timestamp'])
            stats["samples totais"]=len(dataframe)
            #stats["samples eliminados por 1 minuto"]=len(dataframe[dataframe["consume_time"] <= 60])
            
            for nominal_column in nominal_columns:
                if nominal_column in dataframe.columns:
                    dataframe[nominal_column] = dataframe [nominal_column].astype(str)
                    dataframe[nominal_column]  = dataframe[nominal_column].str.replace(" ","")
                    nominal_dataframe=pd.get_dummies(dataframe[nominal_column])
                    dataframe.drop(nominal_column, axis=1, inplace=True)
                    dataframe = pd.concat([dataframe,nominal_dataframe], axis=1)
            
            dataframe.drop(list(dataframe.filter(regex = 'Unna')), axis = 1, inplace=True)
            dataframe.drop(["timestamp"], axis=1, inplace=True)
            dataframe.drop(["timezone"], axis=1, inplace=True)
            if "brand" in dataframe.columns:
                dataframe.drop(["brand"], axis=1, inplace=True)
            if "manufacturer" in dataframe.columns:
                dataframe.drop(["manufacturer"], axis=1, inplace=True)
            if "model" in dataframe.columns:
                dataframe.drop(["model"], axis=1, inplace=True)
            
            print (">>>>>  "+model.split("/")[1].split(".")[0])
            if len(dataframe.index) > (len(dataframe.columns))*100:
                dataframe =  dataframe.sample(n=(round(len(dataframe.columns)*100))-1, random_state=1)  
            print (">>>>>  "+str(len(dataframe)))
            
            target = dataframe["consume_time"]
            dataframe=dataframe.drop("consume_time", axis=1)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(dataframe, target, 
                                                                test_size = 0.2, shuffle=False 
                                                                )
            
            dataframe["consume_time"] = target 
            
            X_train["consume_time"]=y_train
            X_test["consume_time"]=y_test
            upper = medcouple_row.upper.iloc[0]
            lower = medcouple_row.lower.iloc[0]
            stats ["medcouple"] = medcouple_row.medcouple.iloc[0]
            # upper = X_train["consume_time"].quantile(q3) #+(X_train["consume_time"].quantile(q3)-X_train["consume_time"].quantile(q1))
            # lower = X_train["consume_time"].quantile(q1) #-(X_train["consume_time"].quantile(q3)-X_train["consume_time"].quantile(q1))
            #lower =60
            
            # stats["samples eliminados por 1 minuto"]=len(dataframe[dataframe["consume_time"] <= 60])
            stats["samples eliminados por limite superior"]=len(dataframe[dataframe["consume_time"] >= upper])
            stats["samples eliminados por limite inferior"]=len(dataframe[dataframe["consume_time"] <=lower])
            stats ["tempo limite inferior"]=lower
            stats ["tempo limite superior"] = upper
           
            X_train = X_train[X_train["consume_time"] <= upper]
            X_train = X_train[X_train["consume_time"] >= lower]
            X_test = X_test[X_test["consume_time"] <= upper]
            X_test = X_test[X_test["consume_time"] >= lower]
            
            
            y_train = X_train["consume_time"]
            X_train.drop("consume_time", axis=1, inplace=True)
            y_test = X_test["consume_time"]
            X_test.drop("consume_time", axis=1, inplace=True)
            
            X_train =  X_train.sample(n=10000, random_state=1)  
            #print (">>>>>  "+str(len(X_train)))
            
            X_test =  X_test.sample(n=2500, random_state=1)  
            X_test = X_test[X_test.consume_time > 0]
            
            y_train = X_train["consume_time"]
            X_train.drop("consume_time", axis=1, inplace=True)
            y_test = X_test["consume_time"]
            X_test.drop("consume_time", axis=1, inplace=True)
            X_test.to_csv("random_forest_all/"+model.split("/")[1].split(".")[0]+"_X_test.csv", sep=";")

    
            del dataframe
            del target
            if len(X_train.index) > (len(X_train.columns))*10:
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)

                
                #rgr =  DecisionTreeRegressor(**ast.literal_eval(model_list[9]["regressor"]))
                rgr = xg.XGBRegressor(**ast.literal_eval(modelo["regressor"]))

                rgr.fit(X_train, y_train)
                # coef_dict = {}
                # for coef, feat in zip(rgr.coef_,X_train.columns):
                #     coef_dict[feat] = coef
                # print (coef_dict)
                
                y_pred=rgr.predict(X_test)
    
                stats ["test media"] = y_test.mean()
                stats ["test mediana"] = y_test.median()
                stats ["test desvio"] = y_test.std()
                stats ["test minimo"] = y_test.min()
                stats ["test maximo"] = y_test.max()
                from sklearn.metrics import mean_absolute_error
                stats ["mean absolute error"]=mean_absolute_error(y_test,y_pred)
                from sklearn.metrics import  mean_absolute_percentage_error
                stats ["mean_absolute_percentage_error"] = mean_absolute_percentage_error(y_test, y_pred)
                
                
                shap_dict = {"modelo":"",
                      "shap_values":"",
                      "feature_importance":"",
                      "interaction":""
                      }
                
                shap_dict ["modelo"] = stats ["model"]
                explainer = shap.TreeExplainer(rgr, feature_perturbation="tree_path_dependent")
             
                shap_dict["shap_values"] = explainer(X_test, check_additivity = False)

                feature_names = shap_dict["shap_values"].feature_names
                
                shap_df = pd.DataFrame(shap_dict["shap_values"].values, columns=feature_names)
                vals = np.abs(shap_df.values).mean(0)
                shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
                shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
                shap_importance_transpose = shap_importance.T
                shap_importance_transpose, shap_importance_transpose.columns = shap_importance_transpose[1:] , shap_importance_transpose.iloc[0]
                shap_importance_transpose=shap_importance_transpose.rename(index={'feature_importance_vals': shap_dict["modelo"]})
                shap_dict ["feature_importance"]=shap_importance_transpose
                shap_dict ["feature_importance"].insert(0,"base_value", explainer.expected_value) 

                
                
                del X_train
                del X_test
                del y_test
                del y_train
                
                return [shap_dict,stats]
                
from multiprocessing import Pool, Manager
import psutil
if __name__ == "__main__":
    mgr = Manager()
    ns = mgr.Namespace()
    medcouples = pd.read_csv("lower_upper_boundering_medcouple_detection.csv", sep=";")
    ns.medcouples_df = medcouples
    
    model_list = pd.read_csv("xgbooster_all/top_100_best_mape/tunning_xgbooster.csv", sep=";", usecols=["modelo","regressor"]).to_dict('records')
    # model_list = [model_list[0]]
    #model_list = glob.glob("teste/*")
    
    pool = Pool(psutil.cpu_count())
    models_list_dataframe = pool.map(worker_regressor_model, model_list)
    pool.close()
    pool.join()
    models_list_dataframe =[x for x in models_list_dataframe  if x is not None]
    
    shap_df =[]
    pred_df= []
    shap_values_df =[]
    for model in models_list_dataframe:
        if not model is None:
            # model[0]["feature_importance"].reset_index(inplace=True)
            shap_df.append(model[0]["feature_importance"])
        
    for model in models_list_dataframe:
        if not model is None:
            pred_df.append(model[1])
    
    
    shap_df=pd.concat(shap_df)
    pred_df=pd.DataFrame(pred_df)
    
    pred_df.to_excel("xgbooster_all/top_100_best_mape/top_100_best_mape_xgbooster_model_pred_df.xlsx")
    pred_df.to_csv("xgbooster_all/top_100_best_mape/top_100_best_mape_xgbooster_model_pred_df.csv", sep=";")
    shap_df.to_csv("xgbooster_all/top_100_best_mape/top_100_best_mape_xgbooster_model_shap_pred_df.csv", sep=";")
    shap_df.to_excel("xgbooster_all/top_100_best_mape/top_100_best_mape_xgbooster_model_shap_pred_df.xlsx")
    #top_100_best_mape/best_mape
