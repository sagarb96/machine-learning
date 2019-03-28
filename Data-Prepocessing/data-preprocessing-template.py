#Template for date preprocessing

#Importing the libraries - numpy, matplotlib.pyplot & pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the data from Data.csv
dataset = pd.read_csv('Data.csv')
#independent variables
X = dataset.iloc[:, :-1].values
#dependent variable
y = dataset.iloc[:, 3].values

#Taking care of missing data
#Importing "preprocessing" library from Scikit LearnImputer
#   that contains the Imputer class
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',
                  strategy = 'mean', axis=0)
imputer = imputer.fit(X[:, 1:3]) #fitting the imputer object on
                                #column 1 & 2 of X.
X[:, 1:3] = imputer.transform(X[:, 1:3]) #replaces missing data 
                                        #with mean
                                        
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #column 0 is categorical
onehotencoder = OneHotEncoder(categorical_features = [0]) 
X = onehotencoder.fit_transform(X).toarray() #to include dummy columns
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) #dependent variable array is categorical
#Note: Not required to apply OneHotEncoder on dependent variable
#       as model already knows it's categorical and has no order.

