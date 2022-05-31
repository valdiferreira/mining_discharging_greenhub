#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:26:46 2021

@author: Valdi Júnior
"""



import multiprocessing as mp
import glob
import pandas as pd
from multiprocessing import Pool
from ast import literal_eval
from multiprocessing import Manager
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

'''Seleciona apenas os samples que possuem discharging'''
def worker_discharging(file_pool):
    """worker function"""
    
    df=pd.DataFrame()
    file = 'new-dataset-samples-discharging/samples-discharging.query.'+file_pool.split(".")[2] +'.csv'
    if (not glob.glob(file)):
        df=pd.read_csv(file_pool, sep=";")
        df_discharging = df.loc[df['battery_state'] == 'Discharging']
        df_discharging.to_csv(file, mode="w", header=True)
        

'''Cria um arquivo de sample para cada dispositivo'''
def worker_device(device_pool):
    
    files_list_worker = glob.glob("new-dataset-samples-discharging/*.csv")
    files_list = 'new-device-samples-discharging/device-samples-discharging.query.'+str(device_pool) +'.csv'
    df_discharging=pd.DataFrame()       
    while files_list_worker:
        samples_discharging = files_list_worker.pop()
        df = pd.read_csv(samples_discharging)
        if (df_discharging.empty):
            df_discharging = df.loc[df['device_id'] == device_pool]
        else:
            df_discharging = pd.concat([df_discharging, df.loc[df['device_id'] == device_pool]])
    df_discharging.to_csv(files_list, mode="w", header=True, sep =";")
    print (device_pool)


'''Agrupa as 5 primeiras colunas do app_process'''
def worker_group_app_process(file_pool):

    file = "dataset-grouped_app_processes/grouped_sample-app_processes."+file_pool.split(".")[2]+".csv"
    if (not glob.glob(file)):
        df_group = pd.read_csv(file_pool, header=None, sep='\n')
        df_group = df_group[0].str.split(';', expand=True)
        df_group_0 = df_group.groupby(1)[0].apply(list)
        df_group_2 = df_group.groupby(1)[2].apply(list)
        df_group_3 = df_group.groupby(1)[3].apply(list)
        df_group_4 = df_group.groupby(1)[4].apply(list)
        df_group_5 = df_group.groupby(1)[5].apply(list)
        df_group_result = pd.concat([df_group_0, df_group_2], axis=1)
        df_group_result = pd.concat([df_group_result, df_group_3], axis=1)
        df_group_result = pd.concat([df_group_result, df_group_4], axis=1)
        df_group_result = pd.concat([df_group_result, df_group_5], axis=1)        
        df_group_result.to_csv(file)
        print (file_pool.split(".")[2])

"Agrega os processos agrupados aos samples correspondentes"
def worker_device_processes (sample_file):
    if (not glob.glob("dataset-devices-grouped-app_processes/devices-grouped-app_processes."+str(sample_file.split(".")[2])+".csv")):
        summary_chunk = pd.read_csv("summary_app-processes/grouped_summary_processes.all.csv", sep=(";"), usecols=["1","processes_file"], chunksize = 200000)
        sample_df = pd.read_csv(sample_file, sep=(";"), usecols=["id"])
        sample_df=sample_df["id"].astype(int)
        samples_id = set()
        for chunk in (summary_chunk):
            
            chunk = chunk[chunk["1"]!="sample_id"]
            chunk["1"]=chunk["1"].astype(int)
        
            df_chunk=(chunk["processes_file"][chunk["1"].isin(sample_df)])
            samples_id.update(df_chunk.unique())
         
        
        samples_processes_df = pd.DataFrame()
        
        for item in samples_id:
            file = "dataset-grouped_app_processes/"+str(item)+"_grouped_sample-app_processes.csv"
            file = pd.read_csv(file)
            file["1"]=file[file["1"]!="sample_id"]
            file=file.dropna()
            file["1"]=file["1"].astype(int)
            
            if samples_processes_df.empty:
                samples_processes_df = file[file["1"].isin(sample_df)]
            else:
                samples_processes_df=pd.concat([samples_processes_df,file[file["1"].isin(sample_df)]])
        samples_processes_df.to_csv("dataset-devices-grouped-app_processes/devices-grouped-app_processes."+str(sample_file.split(".")[2])+".csv", sep=(";"))

"""Função auxiliar para substituir nomes ausentes dos aplicativos
 pelo nome dos pacotes correspondentes"""
def change_null_for_package(application, package):
    
    application_null = [i for i,val in enumerate(application) if val=="NULL"]
    for index in application_null:
        application[index]=package[index]
    return (application)


"Troca o "
def worker_processes_device_change_null_app_for_package (processes_file):
        
    if (not glob.glob("dataset-transformed-devices-grouped-app_processes/"+processes_file.split("/")[1])):
        dataframe = pd.read_csv(processes_file, sep=";", usecols=["1","2","3","4","5"])
        dataframe["3"] = dataframe["3"].apply(literal_eval)
        dataframe["2"] = dataframe["2"].apply(literal_eval) 
        dataframe["3"]=dataframe.apply(lambda x: change_null_for_package(x['3']
                                                                 , x['2'])
                               , axis=1)
        dataframe=dataframe.drop(list(dataframe.filter(regex = '2')), axis = 1)
        dataframe["4"] = dataframe["4"].apply(literal_eval)
        dataframe["application"]=dataframe.apply(lambda x: [m+"_"+str(n) for m,n in zip(x["3"],x["4"])], axis=1)
        dataframe=dataframe.drop(list(dataframe.filter(regex = '4')), axis = 1)
        dataframe=dataframe.drop(list(dataframe.filter(regex = '3')), axis = 1)  
        dataframe.to_csv("dataset-transformed-devices-grouped-app_processes/"+processes_file.split("/")[1], sep=";")

''' Coloca na mesma linha dois samples sequidos, emparelhando os samples'''
def worker_processes_dummy_process(processes_file):
    mlb = MultiLabelBinarizer()
    dataframe = pd.read_csv(processes_file, sep=";", usecols=["1","application"])    
    dataframe["application"] = dataframe["application"].str.strip('[]').str.split(',')
    processes_dummies = (pd.DataFrame(mlb.fit_transform(dataframe['application']),columns=mlb.classes_, index=dataframe.index))

    dataframe=dataframe.drop(["application"], axis=1)
    dataframe = pd.concat([dataframe,processes_dummies], axis=1)
    dataframe.to_csv("dataset-transformed-dummy-devices-grouped-app_processes/"+processes_file.split("/")[1], sep=";")

"""Reduz o tamanho dos dataframe trocando os tipos dos dados 
parar que sejam mais eficientes"""
def reduce_mem_usage(props):

    NAlist = []

    for col in props.columns:
        
        
        props.mobile_network_type = props.mobile_network_type.astype(str)
        if (props[col].dtype) == object:
            props[col] = props[col].astype('category')
            props[col] = props[col].astype('category')
            
            
        elif not props[col].dtype.name == 'category':

            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
           
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                
                props[col].fillna(0,inplace=True)  
                   
            
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
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
            
            
            else:
                props[col] = props[col].astype(np.float32)
            

    return props

""" 
Cria a variável alvo consume_time
Cria as variáveis independentes user_storage_busy_percent, 
system_size_partition_busy e ram_busy_percent.
Agrega as informações dos dispositivos aos samples

"""

def worker_target_var_and_clean(sample_file):
    if (not glob.glob("csv_dummy_dataset_target_var_time_and_clean/"+(sample_file.split("/")[1]).split(".")[2]+".csv")):
    
        sample_file_df= pd.read_csv(sample_file, sep=";")
        sample_file_df=sample_file_df[sample_file_df.columns[~sample_file_df.columns.str.contains('Unnamed:')]]
        sample_file_df=sample_file_df.sort_values("timestamp", axis=0, ascending=True)
        columns = ["timestamp","battery_level"]
        for column in columns:
            last_column = "last_"+str(column)
            sample_file_df[last_column] = sample_file_df[column].shift(-1)
        sample_file_df["consume_time"] = pd.to_datetime(sample_file_df['last_timestamp'], format="%Y-%m-%d %H:%M:%S")-pd.to_datetime(sample_file_df['timestamp'] , format="%Y-%m-%d %H:%M:%S")
        sample_file_df["consume_time"] = sample_file_df["consume_time"].astype('timedelta64[s]')
                    
        sample_file_df['user_storage_busy_percent'] = (((sample_file_df["total"] - sample_file_df["free"]) / sample_file_df["total"]) * 100).round(0)
        sample_file_df['system_size_partition_busy'] = (sample_file_df["total_system"] - sample_file_df["free_system"])
        sample_file_df['ram_busy_percent'] = (((sample_file_df['memory_active'] + sample_file_df['memory_inactive']) / sample_file_df['memory_free']) * 100).round(0) 
        sample_file_df= sample_file_df.loc[(sample_file_df.battery_level) - (sample_file_df.last_battery_level) < 3]
        sample_file_df= sample_file_df.loc[(sample_file_df.battery_level) - (sample_file_df.last_battery_level) > 0.1]
        sample_file_df=sample_file_df.dropna()
        sample_file_df=sample_file_df.drop(list(sample_file_df.filter(regex = 'Unna')), axis = 1)
        device_id=sample_file.split(".")[2] 
        processes_file="dataset-transformed-dummy-devices-grouped-app_processes/devices-grouped-app_processes."+device_id+".csv"
        if (glob.glob(processes_file)):
              
              sample_file_df = sample_file_df.drop(list(sample_file_df.filter(regex = 'last')), axis = 1)
              df_device_processes = pd.read_csv (processes_file, sep=";")
              if (not df_device_processes.empty):
                  
                    sample_file_df = pd.merge(sample_file_df, df_device_processes
                                  , how="left", left_on="id", right_on="1")
                    
                    
                    device_row = (ns.device_df.loc[ns.device_df['id'] == device_id])
                    sample_file_df ["brand"] = device_row.iloc[0]["brand"]
                    sample_file_df ["model"] = device_row.iloc[0]["model"]
                    sample_file_df ["manufacturer"] = device_row.iloc[0]["manufacturer"]
                    sample_file_df ["os_version"] = device_row.iloc[0]["os_version"]
                    sample_file_df ["is_root"] = device_row.iloc[0]["is_root"]
                   
                    sample_file_df = sample_file_df.drop("1", axis=1)
  
                    sample_file_df=sample_file_df.drop(list(sample_file_df.filter(regex = 'Unna')), axis = 1)
                    sample_file_df=reduce_mem_usage(sample_file_df)
                    if not sample_file_df.empty:
                        sample_file_df.to_csv("csv_dummy_dataset_target_var_time_and_clean/"+(sample_file.split("/")[1]).split(".")[2]+".csv", sep=";")

if __name__ == "__main__":
  
   
    pool = Pool(mp.cpu_count()-1)
    files_list= glob.glob("dataset-samples/*.csv")
    results = pool.map(worker_discharging, files_list)
    #close the pool and wait for the work to finish
    pool.close()
    pool.join()


    files_list = glob.glob("new-dataset-samples-discharging/*.csv")
    for file_discharging in files_list:
        pool = Pool(4)
        df=pd.read_csv(file_discharging, sep=",")
        device_list = df["device_id"]
        device_list=list(set(device_list))
        device_samples_discharging_list=glob.glob("new-device-samples-discharging/*.csv")
        
        for device in device_samples_discharging_list:
            if int(device.split(".")[2]) in device_list:
                    device_list.remove(int(device.split(".")[2]))

        if device_list:
            results = pool.map(worker_device, device_list)
            pool.close()
            pool.join()        
    

    pool = Pool(mp.cpu_count()-1)

    files_list = glob.glob("new-device-samples-discharging/*.csv")
    results = pool.map (worker_device_processes, files_list)

    pool.close()
    pool.join()

    pool = Pool(mp.cpu_count()-1)

    files_list = glob.glob("dataset-devices-grouped-app_processes/*.csv")
    results = pool.map (worker_processes_device_change_null_app_for_package, files_list)
    
    pool.close()
    pool.join()
    
    pool = Pool(mp.cpu_count()-1)

    files_list = glob.glob("dataset-transformed-devices-grouped-app_processes/*.csv")
    results = pool.map (worker_processes_dummy_process, files_list)

    pool.close()
    pool.join()
    
        
    all_filenames = [i for i in glob.glob("dataset-devices/*.csv")]
    device_df = pd.concat([pd.read_csv(f, header=None, sep='\n') for f in all_filenames ])
    device_df = device_df [0].str.split(';', expand=True)
    device_df.columns = device_df.iloc[0]
    device_df=device_df.drop(device_df.index[:1])
    
    mgr = Manager()
    ns = mgr.Namespace()
    ns.device_df = device_df
    
    pool = Pool(mp.cpu_count()-1)

    files_list = glob.glob("dataset-device-samples-discharging/*.csv")
    results = pool.map(worker_target_var_and_clean, files_list)

    pool.close()
    pool.join()
    

