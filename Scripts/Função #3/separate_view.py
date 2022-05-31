#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:46:07 2021

@author: dinho
"""
import pandas as pd
from multiprocessing import Pool
import glob
import psutil


if __name__ == "__main__":
    
    """ Cria uma lista com todos os modelos presentes nos arquivos"""
    models_list=[]
    all_files = glob.glob("devices_top_20_processes_samples_full/*.csv")
    for samples in all_files:
        chunks = pd.read_csv(samples,sep=";", chunksize=2, usecols=["model", "timestamp"])
        for chunk in chunks:
            models_list.append(chunk['model'].iloc[0])
            break
    
    model_list=list(set(models_list))
    from functools import partial
    
    from multiprocessing import Manager
    manager = Manager()
    all_files = glob.glob("ram_devices_top_25_processes_samples_full/*.csv")
    shared_list = manager.list(all_files)
    
    def worker_read_sample_files(sample_file, constant):
        global shared_list
        df = pd.read_csv(sample_file, sep=";")
        
        df = df.loc[df['model'] == constant]
        if (len(df)>0):
            shared_list.remove(sample_file)
            return df
        else:
            return None
    
    """ Cria arquivos com samples separados por modelo"""
    for model in model_list:
        model = model["modelo"].split("/")[1].split(".")[0]
        print (model)
        if not glob.glob ("ram_model_view_devices_100_processes_samples_full/"+model+".csv"):
            
            pool = Pool(psutil.cpu_count())
            parcial_x=partial(worker_read_sample_files, constant=model)
            dataframe = pool.map(parcial_x, shared_list)
            
            dataframe=[x for x in dataframe if x is not None]
            if not len(dataframe)==0:
                dataframe=pd.concat(dataframe, axis=0)
                dataframe=dataframe.fillna(0)
                #constant_list = dataframe.columns[dataframe.nunique() <= 1]
                #dataframe= dataframe.drop(constant_list, axis=1)
                model.replace("/","")
                dataframe.to_csv("ram_model_view_devices_100_processes_samples_full/"+model+".csv", sep=";")
            del dataframe
            pool.close()
            pool.join()
    
