# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:24:51 2023

@author: Rajdeep
"""

import pandas as pd
import numpy as np
import pickle as pkl
import dill


with open('data/datasets/models/pipelinetry.pkl', 'rb') as f:
    pipeline,categories = dill.load(f)
print(pipeline)

with open('data/datasets/models/Model_pipeline.pkl', 'rb') as f:
    dt_model = pkl.load(f)

df = pd.read_csv('data/datasets/testing_dataset/X_test.csv')
print(df.isnull().sum())
print(df.isnull().shape[0])

predictions = dt_model.predict(df)
print(predictions)


y_proba = dt_model.predict_proba(df)*100
y_proba = np.round(y_proba, decimals=0)

fraud_probabilities = y_proba[:, 1] 

print(y_proba)

with open('Models/dt_Model_pipeline.pkl', 'wb') as file:
    pkl.dump(dt_model, file)