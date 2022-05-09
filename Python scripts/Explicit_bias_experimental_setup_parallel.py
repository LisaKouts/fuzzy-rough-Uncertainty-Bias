#!/usr/bin/env python
# coding: utf-8

# In[8]:


import argparse
import math
import numpy as np
import pandas as pd
from numpy import linalg as la
import concurrent.futures
import time
import multiprocessing
from Preprocess_data import preprocess_datasets
from FRU import create_regions, FRU_experimental_setup
from csv import writer

# Import dataset: 'german.data', 'synthetic.csv', 'BROWARD_CLEAN.csv', 'titanic.csv', 'adult.data'
dataset = 'adult.data'
df = preprocess_datasets(dataset).preprocess_dataset()
initial_conditions = FRU_experimental_setup(df).initialize_objects(4)

# Implicators: 'Luka', 'Fodor', 'Godel', 'Goguen'
# Conjunctions: 'Luka', 'Drastic', 'Standard', 'Algebraic'
# dist_measure: 'HMOM', 'HEOM'

conj = 'Luka'
# store distance matrix in binary format
# save hypeparameters in csv

def main_script(hyperparameters, initial):
    full = FRU_experimental_setup(df).fr_regions(hyperparameters[0], hyperparameters[1], conj, hyperparameters[2], initial)
    for col in df.columns[:-1]:
        co=list(df.columns)
        co.remove(col)
        prot = FRU_experimental_setup(df[co]).fr_regions(hyperparameters[0], hyperparameters[1], conj, hyperparameters[2], initial)

        res = [col, FRU_experimental_setup(df).uncertainty(full, prot, 0),
              FRU_experimental_setup(df).uncertainty(full, prot, 1), hyperparameters[0], hyperparameters[1], conj, hyperparameters[2]]
        print(res)

        distance, imp, sm = hyperparameters
        with open(f"results-{distance}-{imp}-{sm}.csv", 'a') as f:
            csv_writer = writer(f, lineterminator='\n')
            csv_writer.writerow(res)
    
if __name__ == '__main__':
    # `hyperparameter_set = ['HMOM','Luka',0.5]
    arg_parser = argparse.ArgumentParser(description='simulation')
    arg_parser.add_argument('--distance', help='distance measure')
    arg_parser.add_argument('--imp', help='implicator')
    arg_parser.add_argument('--sm', type=float, help='smoothing parameter')
    options = arg_parser.parse_args()
    hyperparameter_set = [options.distance, options.imp, options.sm]
    start1 = time.perf_counter()
    main_script(hyperparameter_set, initial_conditions)
    finish1 = time.perf_counter()
    print(f'1st paral Finished in {round(finish1-start1,2)} second(s)')
