# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:19:01 2023

@author: Rajdeep
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle as pkl
import dill


#reading the CSV file
data = pd.read_csv("data/datasets/Telewire Analytics Cell_tower_data.csv")
print(data.head(10))

# getting the info about dataset
print(data.info())

#discription of dataset
print(data.describe().T)

#sahpe of data
print(data.shape)

#Find the duplicates
print(data.duplicated().any())

# Total duplicate rows
data.duplicated().sum()
print(data.duplicated().sum())

#removing the duplicated values
data.drop_duplicates(inplace=True)
print(data.drop_duplicates(inplace=True))

# removing null-values
data.dropna()
print(data.dropna())

#Datatypes
print(data.dtypes)

# changing the datatype of 'maxUE_UL+DL' column
data['maxUE_UL+DL'] = pd.to_numeric(data['maxUE_UL+DL'], errors='coerce')
print(data.isna().sum())


#replacing NaN values with mean 
cols = ['maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL']

for col in cols:
    mean = data[col].mean()
    data[col].fillna(mean, inplace=True)
    
# Loop through all columns in the dataframe
for col in data.columns:
    # Check if the column is of integer data type
    if data[col].dtype == 'float64':
        # Check if there are any non-integer values in the column
        if data[col].apply(lambda x: isinstance(x, str)).sum() > 0:
            # Replace string values with the mean value of the column
            col_mean = data[col].replace({np.nan: None, '': None}).astype(float).mean()
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(col_mean)
            

# checking it's converted or not
print(data.dtypes)

## Check value distribution
#count of 0's and 1's in coloumn in Unusual
print(data['Unusual'].value_counts())

# graphics
y = data['Unusual']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
axes[0].bar(["Usual", "Unusual"],y.value_counts())

for i, v in enumerate(y.value_counts()):
    axes[0].text(i-0.05, v+300, str(v))
    
axes[1].pie(y.value_counts(), labels=["Usual", "Unusual"],autopct='%1.1f%%')
fig.tight_layout()

#histogram of coloumns
data.hist(figsize = (12,12))
plt.show()

#box plot for outlier detection
sns.set(rc={'figure.figsize':(16,10)})
sns.boxplot(data=data)

#Correlation 
print(data.corr())

#Correlation plot
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)

## train test split
data.to_csv("data/datasets/clean_dataset/data.csv",index=False)

# train test split dataset
from sklearn.model_selection import train_test_split
def split_data(input_file, output_path):
    data = pd.read_csv(input_data_path, encoding = "unicode_escape")
    train,test= train_test_split(data, test_size=0.30, random_state=5)
    train.to_csv('data/datasets/training_dataset/train.csv',index=False)
    test.to_csv('data/datasets/testing_dataset/test.csv',index=False)
    
def train_data():
    train1 = pd.read_csv('data/datasets/training_dataset/train.csv', encoding= 'unicode_escape')
    y_train = train1["Unusual"]                     
    X_train = train1.drop(["Unusual"], axis=1)
    return X_train,y_train

def test_data():
    test1 = pd.read_csv('data/datasets/testing_dataset/test.csv', encoding= 'unicode_escape')
    y_test = test1["Unusual"]                     
    X_test = test1.drop(["Unusual"], axis=1)
    return X_test,y_test

input_data_path = open(r"data/datasets/clean_dataset/data.csv")
output_data_path = (r" ")

split_data(input_data_path,output_data_path)

# save train dataset 
X_train,y_train = train_data()
X_train.to_csv('data/datasets/training_dataset/X_train.csv',index=False)
y_train.to_csv('data/datasets/training_dataset/y_train.csv',index=False)

# save train dataset 
X_test,y_test = test_data()
X_test.to_csv('data/datasets/testing_dataset/X_test.csv',index=False)
y_test.to_csv('data/datasets/testing_dataset/y_test.csv',index=False)

print(X_train.head(5))

print(X_train.dtypes)

#categorical dataset
data_new = pd.read_csv('data/datasets/clean_dataset/data.csv',  encoding= 'unicode_escape')
data1 = data_new.drop(["Unusual"], axis=1)
print(data1['CellName'].unique())
# get unique values of column 'B' as a NumPy array
categories_CellName = np.unique(data1['CellName'])
print(print(categories_CellName))
categories_Time = np.unique(data1['Time'])
print(categories_Time)
categories = ['categories_CellName', 'categories_Time']
print(categories)

numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), 
           ("scaler", StandardScaler())])

print(numeric_features)

categorical_features = ['CellName', 'Time']
categorical_transformer = Pipeline(
    steps=[ ("imputer", SimpleImputer(strategy="most_frequent")),
             ("onehot", OneHotEncoder(handle_unknown='ignore'))])

# pipeline
pipelinetry = ColumnTransformer(transformers=[ 
                                                ("num", numeric_transformer, numeric_features),
                                                ("cat", categorical_transformer, categorical_features)
                                                ])

# savepipeline
with open('data/datasets/models/pipelinetry.pkl', 'wb') as f:
    dill.dump((pipelinetry,categories), f)
    
### Decsion tree
with open('data/datasets/models/pipelinetry.pkl', 'rb') as f:
    pipeline,categories = dill.load(f)
print(pipeline)

# Classifier Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,accuracy_score

# Define the model pipeline
model_pipeline = Pipeline([
    ('preprocessing', pipelinetry),
    ('DT_model', DecisionTreeClassifier( criterion= 'entropy', max_depth = None, min_samples_leaf= 2, min_samples_split = 10))    
])

# Fit the model pipeline to the training data
model_pipeline.fit(X_train, y_train)

# predict the values
predict = model_pipeline.predict(X_test)

#accuracy
print(f"accuracy : {accuracy_score(predict,y_test)}")

# recall
print(f"recall score : {recall_score(predict,y_test)}")

print(predict)

# save final model 
with open('data/datasets/models/Model_pipeline.pkl', 'wb') as f:
    pkl.dump(model_pipeline, f)
    
print('Model has been saved and terminated code!')