#Simple Linear Regression

#Importing the libraries - numpy, matplotlib.pyplot & pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the data from Data.csv
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #matrix
y = dataset.iloc[:, 1].values   #vector


#Splitting the dataset into the Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
                                                    random_state = 0)

#Feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''


#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


