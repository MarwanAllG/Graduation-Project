from google_trans_new import google_translator 
from googletrans import Translator
import pandas as pd
import re
df = pd.read_excel('CarS.xlsx',sheet_name='S')
df = pd.read_excel('new.xlsx')

# Count the null values
df.isna().sum().sum()
# Removing any nulls
df = df[df.brand.notnull()]
df = df[df.transmission.notnull()]
df = df[df.color.notnull()]
df = df[df.fuelType.notnull()]
df = df[df.model.notnull()]
df = df[df.year.notnull()]
df = df[df.mileage.notnull()]
df = df[df.location.notnull()]
df = df[df.engineSize.notnull()]
df = df[df.price.notnull()]
# Transfrom to lowercase
df['brand'] = df['brand'].str.lower()
df['model'] = df['model'].str.lower()
df['color'] = df['color'].str.lower()
df['transmission'] = df['transmission'].str.lower()
df['fuelType'] = df['fuelType'].str.lower()
df['location'] = df['location'].str.lower()
df['color'].value_counts()

# Extract the location 
df['city'] = df['location'].str.extract('(,\s\w+)')
df['city'] = df['city'].str.replace('(,\s)','')
df.drop(['location'],axis = 1,inplace = True)
df.drop(['shape'],axis = 1,inplace = True)
# Cleaning the fuel type column
df['fuelType'] = df['fuelType'].str.replace('(petrol)','gas')
# Cleaning the engine size
df['engineSize'] = df['engineSize'].str.replace('(L)','')
df.to_excel('newCarSwitch.xlsx', index = False)