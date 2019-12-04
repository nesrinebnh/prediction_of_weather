import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import csv
import datetime
from math import sqrt
from sklearn.svm import SVR
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from tkinter import *
warnings.filterwarnings("ignore", category=DeprecationWarning) 
def make_numeric_values(arr, title):
    new_arr = []
    for date in arr[title]:
        new_date = make_date(date)
        new_arr.append(new_date)
    arr[title] = new_arr

def fix_array(arr):
    for name in ['Fran Datum Tid (UTC)', 'till', 'day']:
        make_numeric_values(arr, name)

def make_date(date):
    new_date = date.split(' ')
    new_date = new_date[0]
    new_date = new_date.split('-')
    new_number = ''
    first = True
    for number in new_date:
        if first:
            first = False
        else:
            new_number = new_number + number
    return new_number

def train():

    ##recuperer dataset
    dataset_url1 = 'smhi-opendata_2_71420_corrected-archive_2019-05-01_14-00-00.csv'
    dataset_url2 = 'smhi-opendata_2_71420_latest-months_2019-05-29_05-00-00.csv'

    ##lire les dataset 
    data1 = pd.read_csv(dataset_url1, sep=';', skiprows=3607, names=['Fran Datum Tid (UTC)', 'till', 'day', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'])
    data2 = pd.read_csv(dataset_url2, sep=';', skiprows=15, names=['Fran Datum Tid (UTC)', 'till', 'day', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'])
    #tracer l'histogramme
    plt.hist(data1["temperature"])
    plt.show()
    data1 = data2.append(data1)
    #eliminer les données inutiles et garder que les dates dans X
    data1 = data1.drop('Tidsutsnitt:', axis=1)
    X = data1.drop(["temperature"], axis=1)
    X = X.drop(['Kvalitet'], axis = 1)
    X = X.drop(['Unnamed: 5'], axis = 1)
    fix_array(X)
    ##garder la temperture dans Y
    y = data1['temperature']
    ##séparer les data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    ##lancer le modele
    tree_model = DecisionTreeRegressor()
    ##alimenter le modele
    tree_model.fit(X_train, y_train)
    ##calculer le % d'erreur
    predictions = tree_model.predict(X_test)
    mean_squared_error(predictions, y_test)

    print("training is over")

def predictWeather():


    ##predict weather
    tree_model = joblib.load('weather_predictor.pkl')

    print("-" * 48)
    print("Enter the details of the date you would like to predict")
    print("\n")
    option = input("Year: ")
    year = option
    option = input("Month number (00): ")
    month = option
    option = input("Day number (00): ")
    theday = option

    day = str(month) + str(theday)

    date = [
        [day, 
        (str(int(day) + 1)), 
        (day)]
    ]
    temp = tree_model.predict(date)[0]
    return temp
temp = predictWeather()
if(temp>16):
    print("sunny day")
else:
    print("runny day")



