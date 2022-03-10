#!/usr/bin/env python
# coding: utf-8

# In[8]:


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
dataset = 'titanic.csv'
df = preprocess_datasets(dataset).preprocess_dataset()
initial_conditions = FRU_experimental_setup(df).initialize_objects(4)

# Implicators: 'Luka', 'Fodor', 'Godel', 'Goguen'
# Conjunctions: 'Luka', 'Drastic', 'Standard', 'Algebraic'
# dist_measure: 'HMOM', 'HEOM'

if __name__ == '__main__':
    open("results.csv", 'a')
    for impli in ['Luka', 'Fodor']:
        for dist_measure in ['HMOM', 'HEOM']:
            for sm in [0.1, 0.3, 0.5, 0.7, 0.9]:
                conj = 'Luka'
                full = FRU_experimental_setup(df).fr_regions(dist_measure, impli, conj, sm, initial_conditions)

                for col in df.columns[:-1]:
                    co=list(df.columns)
                    co.remove(col)
                    prot = FRU_experimental_setup(df[co]).fr_regions(dist_measure, impli, conj, sm, initial_conditions)

                    res = [col, FRU_experimental_setup(df).uncertainty(full, prot, 0),
                          FRU_experimental_setup(df).uncertainty(full, prot, 1), dist_measure, impli, conj, sm]
                    print(res)

                    with open("results.csv", 'a') as f:
                        csv_writer = writer(f, lineterminator='\n')
                        csv_writer.writerow(res)
