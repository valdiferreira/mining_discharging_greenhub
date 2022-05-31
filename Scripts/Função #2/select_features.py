#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 04:55:21 2021

@author: dinho
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 13:10:53 2021

@author: dinho
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool
import glob
import psutil

def reduce_mem_usage(props):
    
    
     
#     start_mem_usg = props.memory_usage().sum() / 1024**2 
#     print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:
        
        if (props[col].dtype) == object:
            props[col] = props[col].astype('category')
            
            
        elif not props[col].dtype.name == 'category':

            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                # columns have missing values filled with 'np.uint8(0)': ")
                props[col].fillna(np.uint8(0),inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            
    return props


"""
Seleciona dentre as variáveis independentes de configuração desejadas desejadas

"""
columns_config = ["consume_time", "voltage", "temperature", "usage","ram_busy_percent" ,
            "system_size_partition_busy", "up_time","user_storage_busy_percent",
            "screen_on", "roaming_enabled", 
            "bluetooth_enabled","location_enabled",
            "power_saver_enabled", "nfc_enabled", "health" ,
            "developer_mode", 'battery_level', "timestamp"
            , "timezone", "country_code", "model", "brand", "manufacturer"
            , "os_version", "screen_brightness", "network_status"]


"""
Seleciona dentre as variáveis independentes de processos de aplicativos desejadas desejadas

"""
no_system_processes=pd.read_csv("no_system_processes.csv", sep=";")
no_system_processes=no_system_processes.T.head(25).index.to_list()

"""
Seleciona dentre as variáveis independentes de processos do sistema desejadas desejadas

"""
system_processes=pd.read_csv("system_processes.csv", sep=";")
system_processes=system_processes.T.head(25).index.to_list()


cols = columns_config + no_system_processes + system_processes
def and_binary_same_column(columns):
    if all(y==1 for y in list(columns)):
        return 1
    else:
        return 0

""" Seleciona apenas os samples que possuem ao menos um processo do sistema e um de aplicativo"""
df_nan = pd.DataFrame()
def worker_read_sample_files(sample_file):
    df = pd.read_csv(sample_file, sep=";")
    df_nan.append(df[df.isna().any(axis=1)])
    #fazer um if checando se o model tem processo e se o dataframe do modelo não ta vazio
    #tem backup em algum lugar, acho que no drive
    
    df=df.dropna()
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.strip("'")
    df = df[[col for col in cols if col in df.columns]]
    if (not df.filter(regex=("_1")).empty and not df.filter(regex=("_0")).empty):
       
        df=df.loc[:, ~df.columns.duplicated()]
        reduce_mem_usage(df).to_csv("ram_devices_top_25_processes_samples_full/"+(sample_file.split("/")[1]).split(".")[0]+".csv", sep =";")
        
if __name__ == "__main__":

    all_files = glob.glob("csv_dummy_dataset_target_var_time_and_clean/*.csv")
    
    pool = Pool(psutil.cpu_count())
    dataframe = pool.map(worker_read_sample_files, all_files)
    
    pool.close()
    pool.join()


