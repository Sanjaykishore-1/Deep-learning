
## Import libraries
==================================

import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout

import warnings
warnings.filterwarnings("ignore")

=====================================

## import data

diabetes = pd.read_csv("diabetes.csv")

=====================================

diabetes.shape

diabetes.isna().sum()

diabetes.dtypes

====================================

x = diabetes.drop('Outcome',axis = 1)
y = diabetes_data[['Outcome']]

===================================

model = Sequential()
model.add(Dense(12, kernel_initializer ="unifrom",activation= "relu"))
model.add(Dense(8, kernel_initializer ="unifrom",activation= "relu"))
model.add(Dense(1, kernel_initializer ="unifrom",activation= "softmax"))





























