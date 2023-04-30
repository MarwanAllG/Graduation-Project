import pandas as pd
import re

df = pd.read_excel('Sayrah.xlsx',sheet_name = '')

df.isnull().sum(axis = 0)
# Removing any nulls
df = df[df.price.notnull()]
df = df[df.color.notnull()]
df = df[df.engine_size.notnull()]
# Lowercase
df['brand'] = df['brand'].str.lower()
df['model'] = df['model'].str.lower()
df['color'] = df['color'].str.lower()
df['transmission'] = df['transmission'].str.lower()
df['fuel'] = df['fuel'].str.lower()
df['location'] = df['location'].str.lower()
# Replace 
df['fuel'].value_counts()
df['fuel'] = df['fuel'].str.replace('(solar)','gas')

df.to_excel('Sayrah.xlsx', index = False)



