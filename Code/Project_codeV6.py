# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold,StratifiedKFold,LeaveOneOut, cross_val_score
# Deep Learning Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tensorflow import keras
import timeit
import pickle as pk
# Importing the data
dataset = pd.read_csv('Project_data6.csv')
# Drop any dupicates
dataset.drop_duplicates()
# Remove any car that has a Mileage value more than 900000
dataset = dataset[dataset["Mileage"] < 900000]
# Create a new attribute called Age from the year column
dataset['Age'] = 2023 - dataset['Year']
#dataset['Color'].value_counts()
#dataset['Color'] = dataset['Color'].str.replace('other','white')
sc1= StandardScaler()
dataset['Mileage'] = sc1.fit_transform(dataset[['Mileage']])
# Show the correlation between the price attribute and the other attributes
#corr = dataset.corr()
#corr.sort_values(["Price"], ascending = False, inplace = True)
#print(corr.Price)
# Splitting all data with all 9 brands to dependent and indpendent variables 
x = dataset[['Age','engineSize','Mileage','Brand','Color','gearType','Region',
             'Model','fuelType']]
x = pd.get_dummies(x, columns=['Brand','Color', 'gearType','Region',
                                 'Model','fuelType'])
y = dataset[['Price']].values
x = x.values

# Here we will split the data into 9 diffrent dataframes based on the brand name and
# also we will split each dataframe to dpendent and indpendent variables
#------------------------------------------------------------------------------
# 1
## Toyota
toyota = dataset[dataset['Brand'] == 'toyota']
xt = toyota[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xt = pd.get_dummies(xt, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yt = toyota[['Price']]
xt = xt.values
#---------------------------
# 2
## Nissan
nissan = dataset[dataset['Brand'] == 'nissan']
xn = nissan[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xn = pd.get_dummies(xn, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yn = nissan[['Price']]
xn = xn.values
#--------------------------
# 3
## GMC
gmc = dataset[dataset['Brand'] == 'gmc']
xg = gmc[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xg = pd.get_dummies(xg, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yg = gmc[['Price']]
xg = xg.values
#----------------------------
# 4
## Mercedes
mercedes = dataset[dataset['Brand'] == 'mercedes']
xm = mercedes[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xm = pd.get_dummies(xm, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
ym = mercedes[['Price']]
xm = xm.values
#----------------------------
# 5
## Kia
kia = dataset[dataset['Brand'] == 'kia']
xk = kia[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xk = pd.get_dummies(xk, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yk = kia[['Price']]
xk = xk.values
#----------------------------
# 6
## Chevrolet
chevrolet = dataset[dataset['Brand'] == 'chevrolet']
xc = chevrolet[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xc = pd.get_dummies(xc, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yc = chevrolet[['Price']]
xc = xc.values
#----------------------------
# 7
## Hyundai
hyundai = dataset[dataset['Brand'] == 'hyundai']
xh = hyundai[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xh = pd.get_dummies(xh, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yh = hyundai[['Price']]
xh = xh.values
#----------------------------
# 8
## Ford
ford =  dataset[dataset['Brand'] == 'ford']
xf = ford[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xf = pd.get_dummies(xf, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yf = ford[['Price']]
xf = xf.values
#-----------------------------
# 9
## Lexus
lexus = dataset[dataset['Brand'] == 'lexus']
xl = lexus[['Age','engineSize','Mileage','Color','gearType','Region',
             'Model','fuelType']]
xl = pd.get_dummies(xl, columns=['Color', 'gearType','Region',
                                 'Model','fuelType'])
yl = lexus[['Price']]
xl = xl.values
#------------------------------------------------------------------------------

# Here are the models we want to test, 9 models
regressors = [['DecisionTreeRegressor',DecisionTreeRegressor(criterion = 'squared_error',random_state=42,max_depth=15)],
              ['XGBRegressor', XGBRegressor(n_estimators=495, max_depth=7, eta=0.1,subsample=0.9, colsample_bytree=0.7)],
              ['GradientBoostingRegressor',GradientBoostingRegressor(n_estimators=600,random_state=42)],
              ['RandomForestRegressor', RandomForestRegressor(n_estimators = 85, criterion = 'squared_error',random_state=42)],
              ['LinearRegression',LinearRegression()],
              ['ExtraTreesRegressor',ExtraTreesRegressor(n_estimators = 47, criterion = 'squared_error',random_state=42)],
              ['CatBoostRegressor',CatBoostRegressor(iterations= 108, random_state= 42)],
              ['SVR',SVR(kernel = 'poly',C=300,gamma=0.15)],
              ['KNN',KNeighborsRegressor(n_neighbors=4,algorithm='auto',weights='distance',metric='manhattan')]]
# Lists of the dependent and indpendent variables of each dataframe, we will use
# this list for the next nested for loop
dfs = [[x,y],[xt,yt],[xn,yn],[xg,yg],[xm,ym],[xk,yk],
                      [xc,yc],[xh,yh],[xf,yf],[xl,yl]]
# List of the names of the accuracy tables we will create to store the results
Accs = ['Accx','Acct','Accn','Accg','Accm','Acck','Accc','Acch','Accf','Accl']
# Using this for loop, we will create an accuracy table for each dataframe to store
# the accuracy of each model tested on a dataframe
for a in Accs:
    vars()[a] = pd.DataFrame(index=None, columns=['Model','MAPE',
                                            'MAE',
                                            'R2'])
# Using this nested for loop we will test each model on each dataframe and store the results on
# the accuracy tables 
start = timeit.default_timer()
for d,a in zip(dfs,Accs):
    x_train, x_test, y_train, y_test = train_test_split(d[0],d[1],test_size=0.20,random_state=42)
    for mod in regressors:
          name = mod[0]
          model = mod[1]
          model.fit(x_train,y_train)
          y_pred = model.predict(x_test)
          MAPE = mean_absolute_percentage_error(y_test, y_pred)
          MAE = mean_absolute_error(y_test, y_pred)
          R2 = r2_score(y_test, y_pred)
          vars()[a] = vars()[a].append(pd.Series({'Model':name, 'MAPE': MAPE,
                                'MAE':MAE,
                                'R2':R2}),ignore_index=True )
          vars()[a] = round(vars()[a],3)
stop = timeit.default_timer() 
# It took about 2 min to run 
print('Time: ', stop - start)       
# This function is for testing each model alone and show the required results fast         
def Models(models):
    
    model = models
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_total = model.predict(x)
    
    print("\t\tError Table")
    print('Mean Absolute Error  : ', mean_absolute_error(y_test, y_pred))
    print('mean absolute percentage error  : ',mean_absolute_percentage_error(y_test, y_pred))
    print('Accuracy on Testing set  : ', r2_score(y_test, y_pred))
    #return y_total, y
    
# Testing each machine learning model alone using the models funcion
Models(ExtraTreesRegressor(n_estimators = 47, criterion = 'squared_error',random_state=42))
Models(XGBRegressor(n_estimators=495, max_depth=7, eta=0.1,subsample=0.9, colsample_bytree=0.7))
Models(GradientBoostingRegressor(n_estimators=600,random_state=42))
Models(DecisionTreeRegressor(criterion = 'squared_error',random_state=42,max_depth=15))
Models(RandomForestRegressor(n_estimators = 85, criterion = 'squared_error',random_state=42))
Models(CatBoostRegressor(iterations= 108, random_state= 42))
Models(KNeighborsRegressor(n_neighbors=4,algorithm='auto',weights='distance',metric='manhattan'))
Models(SVR(kernel = 'poly',C=300,gamma=0.15))
#------------------------------------------------------------------------------


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
regressor = ExtraTreesRegressor(n_estimators = 47, criterion = 'squared_error',random_state=42)
regressor.fit(x_train, y_train) # fit the model
y_pred = regressor.predict(x_test)
r2_score(y_test, y_pred)
pk.dump(regressor, open('regressor.pkl', 'wb'))

plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('Extree Trees Regressor')
plt.axis('equal')
plt.show()

# Deep Learning
x_train, x_test, y_train, y_test = train_test_split(xl, yl, test_size=0.20,random_state=42)
# Function to create the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=33, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics =['mae'])
    return model
model = create_model()
model.summary()
# fit the model
history = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=250, batch_size=32)
# Measure the accuracy
prediction = model.predict(x_test)
print(r2_score(y_test, prediction),' ',mean_absolute_error(y_test, prediction),' ',
      mean_absolute_percentage_error(y_test, prediction))
r2_score(y_test, prediction)
mean_absolute_error(y_test, prediction)
mean_absolute_percentage_error(y_test, prediction)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

prediction = model.predict(x_test)
r2_score(y_test, prediction)
mean_absolute_error(y_test, prediction)
mean_absolute_percentage_error(y_test, prediction)