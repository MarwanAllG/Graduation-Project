import pandas as pd
import re
dataset = pd.read_excel('Project_Data (version 1).xlsb.xlsx',sheet_name='Syrah+Yallah 9 brands only')


# Removing any nulls
dataset = dataset[dataset.transmission.notnull()]
dataset = dataset[dataset.color.notnull()]
dataset = dataset[dataset.fuel_Type.notnull()]
dataset = dataset[dataset.model.notnull()]

# Extracting the engine size
dataset["engine_size"] = dataset['name'].str.extract('(\d\.\d)')
# Extracting the brand name
dataset['brand'] = dataset['name'].str.extract('([A-Z]\w{0,})')
# To extract the model name, we will remove the brand  and engine size and
#the year from the name column and keep only the model name

dataset['name'] = dataset['name'].str.replace(r'(^\w+)','') # Removing brand
dataset['name'] = dataset['name'].str.replace(r'(\d\.\dL|\d\.\d)','') # Removing the engine size
dataset['name'] = dataset['name'].str.replace(r'(\d\d\d\d)','') # Removing the year

dataset['model'] = dataset['name']
dataset.drop(['name'],axis = 1,inplace = True)

dataset['model'].nunique()

# Cleaning the model column
dataset['model'] = dataset['model'].str.replace(r'(\dWD)','')
dataset['model'] = dataset['model'].str.replace(r'([A-Z]WD)','')
dataset['model'] = dataset['model'].str.replace(r'(\d\sDoor)','')
dataset['model'] = dataset['model'].str.replace(r'(\d\sdoor)','')
dataset['model'] = dataset['model'].str.replace(r'(\d\d\d\sHP)','')
dataset['model'] = dataset['model'].str.replace(r'(\d\d\dHP)','')
dataset['model'] = dataset['model'].str.replace(r'(\dX\d)','')
dataset['model'] = dataset['model'].str.replace(r'\([^)]*\)','')
dataset['model'] = dataset['model'].str.replace(r'(\w+\sOption)','')
dataset['model'] = dataset['model'].str.replace(r'(\w+\soption)','')
dataset['model'] = dataset['model'].str.replace(r'(Diesel)','')
dataset['model'] = dataset['model'].str.replace(r'(Hybrid)','')
dataset['model'] = dataset['model'].str.replace(r'(\dx\d)','')
dataset['model'] = dataset['model'].str.replace(r'(\w+\sWheel\sDrive)','')
dataset['model'] = dataset['model'].str.replace(r'(Moonroof)','')
dataset['model'] = dataset['model'].str.replace(r'(Sunroof)','')
# lowercase all the values
dataset['brand'] = dataset['brand'].str.lower()
dataset['model'] = dataset['model'].str.lower()
dataset['color'] = dataset['color'].str.lower()
dataset['transmission'] = dataset['transmission'].str.lower()
dataset['fuel_Type'] = dataset['fuel_Type'].str.lower()
dataset['location'] = dataset['location'].str.lower()
#dataset['model'] = dataset['name'].str.extract(r'(?:(\s\w+\s\w+\s|\s\w+|\s\w+\s\w+\s\w+\s))')
# r'(?:(\s[A-Z]\w{0,}\s[A-Z]\w{0,}\s|[A-Z]\w{0,}))

dataset['Brand'].value_counts()
dataset.isnull().sum()
# Fill in missing engine_size values

# Saving the file
dataset.to_csv('yallamotor_new.csv', index = False)

