#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 07:57:16 2021

@author: dinho
"""
import time
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import shap
import glob
import ast
import pickle



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
start_time = time.time()
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
        dataframe.columns = dataframe.columns.astype(str)
        stats ["model"] = model.split("/")[1].split(".")[0]
        
        
        
        if len(dataframe.index) > (len(dataframe.columns)-1)*10:
            dataframe=dataframe.sort_values(by=['timestamp'])
            stats["samples totais"]=len(dataframe)
            #stats["samples eliminados por 1 minuto"]=len(dataframe[dataframe["consume_time"] <= 60])
            
            for nominal_column in nominal_columns:
                if nominal_column in dataframe.columns:
                    dataframe[nominal_column] = dataframe [nominal_column].astype(str)
                    dataframe[nominal_column]  = dataframe[nominal_column].str.replace(" ","")
                    nominal_dataframe=pd.get_dummies(dataframe[nominal_column], prefix=nominal_column)
                    dataframe.drop(nominal_column, axis=1, inplace=True)
                    dataframe = pd.concat([dataframe,nominal_dataframe], axis=1)
            
            dataframe.drop(list(dataframe.filter(regex = 'Unna')), axis = 1, inplace=True)
            
            dataframe.drop(["timezone"], axis=1, inplace=True)
            dataframe.drop(["timestamp"], axis=1, inplace=True)
            if "brand" in dataframe.columns:
                dataframe.drop(["brand"], axis=1, inplace=True)
            if "manufacturer" in dataframe.columns:
                dataframe.drop(["manufacturer"], axis=1, inplace=True)
            if "model" in dataframe.columns:
                dataframe.drop(["model"], axis=1, inplace=True)
            if "country_code" in dataframe.columns:
                dataframe.drop(["country_code"], axis=1, inplace=True)

            
            
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
            
            X_train =  X_train.sample(n=10000, random_state=1)  
            #print (">>>>>  "+str(len(X_train)))
            
            X_test =  X_test.sample(n=2500, random_state=1)  
            X_test = X_test[X_test.consume_time > 0]
            y_train = X_train["consume_time"]
            X_train.drop("consume_time", axis=1, inplace=True)
            y_test = X_test["consume_time"]
            X_test.drop("consume_time", axis=1, inplace=True)
            #X_test.to_csv("decision_tree_all/"+model.split("/")[1].split(".")[0]+"_X_test.csv", sep=";")

            del dataframe
            del target
            if len(X_train.index) > (len(X_train.columns))*10:
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)

                
                rgr = DecisionTreeRegressor(**ast.literal_eval(modelo["regressor"]))

                rgr.fit(X_train, y_train)

                print ("model")
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
                      "feature_importance_signal"
                      "interaction":""
                      }
                
                shap_dict ["modelo"] = stats ["model"]
                explainer = shap.TreeExplainer(rgr, feature_perturbation="tree_path_dependent")
                explainer.expected_value = explainer.expected_value[0]
                
                shap_dict["shap_values"] = explainer(X_test, check_additivity = False)
                with open("decision_tree_all/"+stats["model"]+".pkl", 'wb') as f:
                    pickle.dump(shap_dict["shap_values"], f)
                f.close()
                
                
                
                del X_train
                del X_test
                del y_test
                del y_train
                
                return [shap_dict,stats]
                

from multiprocessing import Pool, Manager
import multiprocessing
import psutil
print(multiprocessing.__version__)

if __name__ == "__main__":
    mgr = Manager()
    ns = mgr.Namespace()
    medcouples = pd.read_csv("lower_upper_boundering_medcouple_detection.csv", sep=";")
    ns.medcouples_df = medcouples
    
    df_tunning_list=pd.read_csv("new_decision_tree_top_100_sample.csv", sep=";")
    
    df_tunning_list = df_tunning_list.to_dict("records")
    
    
    pool = Pool(psutil.cpu_count())
    models_list_dataframe = pool.map(worker_regressor_model, df_tunning_list)
    pool.close()
    pool.join()
    models_list_dataframe =[x for x in models_list_dataframe  if x is not None]
    
    shap_df=[]
    pred_df= []
    
    for model in models_list_dataframe:
        if not model is None:
            
            shap_df.append(model[0]["feature_importance"])
    
    for model in models_list_dataframe:
        if not model is None:
            pred_df.append(model[1])
    
    
    shap_df=pd.concat(shap_df)
    pred_df=pd.DataFrame(pred_df)
    
    pred_df.to_excel("decision_tree_all/raw_top_100_samples_decision_tree_model_pred_df.xlsx")
    pred_df.to_csv("decision_tree_all/raw_top_100_samples_decision_tree_model_pred_df.csv", sep=";")
    shap_df.to_csv("decision_tree_all/raw_top_100_samples_decision_tree_model_shap_df.csv", sep=";")
    shap_df.to_excel("decision_tree_all/raw_top_100_samples_decision_tree_model_shap_df.xlsx")
    
    print("--- %s seconds ---" % (time.time() - start_time))
