import pandas as pd
import pandasql as ps
import numpy as np



df = pd.read_excel('Project_Data3.xlsx', sheet_name= 'All')

df['Model'] = df['Model'].str.strip()

df['Engine_Size'] = df['Engine_Size'].fillna(df.groupby('Model')['Engine_Size'].transform('median'))
df['Engine_Size'] = df['Engine_Size'].fillna(df.groupby('Model').Engine_Size.transform(func = 'median'))

df.isnull().sum()

med = df.groupby('Model').Engine_Size.agg(func = 'median')


df.to_csv('Project_data4.csv', index = False)


