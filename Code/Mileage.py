import pandas as pd
import pandasql as ps
import numpy as np

df = pd.read_csv('Project_data4.csv')
df1 = pd.read_csv('under10000.csv')

df.isnull().sum()

df1 = df[(df['Mileage'] < 10000) & (df['Mileage'] >= 1000) & (df['Year'] <= 2013)]
df2 = df[(df['Mileage'] < 1000) & (df['Mileage'] >= 100) & (df['Year'] <= 2013)]
df3 = df[(df['Mileage'] < 100) & (df['Year'] <= 2013)]
df4 = df[(df['Mileage'] >= 10000) | (df['Year'] > 2013)]


df1['Mileage']  = df1['Mileage'].apply('{:0<6}'.format)
df2['Mileage']  = df2['Mileage'].apply('{:0<6}'.format)
df3['Mileage']  = df3['Mileage'].apply('{:0<6}'.format)

df1.dtypes
df2.dtypes
df3.dtypes
df4.dtypes

df2['Mileage'] = df2['Mileage'].apply(pd.to_numeric)
df3['Mileage'] = df3['Mileage'].apply(pd.to_numeric)

frame = [df1, df2, df3, df4]

df_clean = pd.concat(frame)

df_clean.to_csv('Project_data5.csv', index = False)