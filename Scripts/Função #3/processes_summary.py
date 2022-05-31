#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:11:07 2022

@author: dinho
"""

import pandas as pd
import glob
from multiprocessing import Pool, Value
import psutil

not_system_process_list = ['_0']  # lista de colunas de processos que não são do sistema
usecols = lambda x: any(substr in x for substr in not_system_process_list)
counter = Value('i', 0)
def worker_value_counts(sample_file):  
    df = (pd.read_csv(sample_file, sep=";", usecols=usecols))
    counter.value += len(df.index)
    return (df.apply(pd.value_counts).dropna(how="all").fillna(0).astype(int))

all_files = glob.glob("csv_dummy_dataset_target_var_time_and_clean/*.csv")
pool = Pool(psutil.cpu_count()-1)
#dataframe = pool.map(worker_value_counts, all_files)
dataframe = pd.concat(pool.map(worker_value_counts, all_files), axis= 1)
pool.close()
pool.join()

dataframe.columns = map(str.lower, dataframe.columns)
no_system_processes=dataframe=(dataframe.groupby(level=0, axis=1).sum().sort_values(by = 1, axis = 1, ascending = False))
no_system_processes.columns = map(str.lower, no_system_processes.columns)
no_system_processes.columns = no_system_processes.columns.str.strip()
no_system_processes.columns = no_system_processes.columns.str.strip("'")
no_system_processes=(no_system_processes.groupby(level=0, axis=1).sum().sort_values(by = 1, axis = 1, ascending = False))
no_system_processes.to_csv("no_system_processes.csv", sep=";")

system_process_list = ['_1']  # lista de colunas de processos que são do sistema
usecols = lambda x: any(substr in x for substr in system_process_list)
counter = Value('i', 0)
def worker_value_counts(sample_file):  
    df = (pd.read_csv(sample_file, sep=";", usecols=usecols))
    counter.value += len(df.index)
    return (df.apply(pd.value_counts).dropna(how="all").fillna(0).astype(int))

all_files = glob.glob("csv_dummy_dataset_target_var_time_and_clean/*.csv")
pool = Pool(psutil.cpu_count()-1)
#dataframe = pool.map(worker_value_counts, all_files)
dataframe = pd.concat(pool.map(worker_value_counts, all_files), axis= 1)
pool.close()
pool.join()

dataframe.columns = map(str.lower, dataframe.columns)
no_system_processes=dataframe=(dataframe.groupby(level=0, axis=1).sum().sort_values(by = 1, axis = 1, ascending = False))
no_system_processes.columns = map(str.lower, no_system_processes.columns)
no_system_processes.columns = no_system_processes.columns.str.strip()
no_system_processes.columns = no_system_processes.columns.str.strip("'")
no_system_processes=(no_system_processes.groupby(level=0, axis=1).sum().sort_values(by = 1, axis = 1, ascending = False))
no_system_processes.to_csv("system_processes.csv", sep=";")









