from numpy import False_
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

path_to_file ="D:/projet m1/telecom_churn_data.csv"
df=pd.read_csv(path_to_file)

#supression des doublons
df.drop_duplicates(keep = 'first', inplace=True) 


#supression des colonnes avec plus de 10% de valeurs manquantes
column_with_nan = df.columns[df.isnull().any()]
for column in column_with_nan:
    if df[column].isnull().sum()*100/df.shape[0] > 10:
        df.drop(column,1, inplace=True)


#supression des colonnes non numerique
for column in df:
    if(df[column].dtypes == 'object'):
       df.drop(column,1, inplace=True)       


#remplacement des valeurs manquantes par 0 pour les variables binaires et la mediane pour les autres
column_with_nan = df.columns[df.isnull().any()]
for column in column_with_nan:
    row=0
    t=True
    while row<1000 and t==True :
        if df[column][row] != 0 and df[column][row] != 1 and pd.isna(df[column][row]) == True:
            t=False
        row+=1
    if t==True:
        df[column]=df[column].fillna(0)
    else:
        df[column]=df[column].fillna(df[column].median())

#detection des valeurs aberrantes et supression des lignes
for column in df:
   Q1= np.percentile(df[column], 25,interpolation = 'midpoint')
   Q3 = np.percentile(df[column], 75,interpolation = 'midpoint')
   IQR = Q1 -Q3
   upper = np.where(df[column] >= (Q3+1.5*IQR))
   lower = np.where(df[column] <= (Q1-1.5*IQR))
   df.drop(upper[0],0, inplace = True)
   df.drop(lower[0],0, inplace = True)