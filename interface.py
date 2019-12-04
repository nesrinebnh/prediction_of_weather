from tkinter import * 
import tkinter as tk
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
    ##tracer l'histogramme
    #plt.hist(data1["temperature"])
    #plt.show()
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
    #predictions = tree_model.predict(X_test)
    #mean_squared_error(predictions, y_test)

    print("training is over")

def predictWeather(year,month,theday):


    ##predict weather
    tree_model = joblib.load('weather_predictor.pkl')

    day = str(month) + str(theday)

    date = [
        [day, 
        (str(int(day) + 1)), 
        (day)]
    ]
    temp = tree_model.predict(date)[0]
    return temp



def show_entry_fields():
	temp = predictWeather(year_input.get(),month_input.get(),day_input.get())
	if(temp >16):
		label_day = Label(right_frame, text="it's a sunny day",font=("Arial",14),bg='#619ecb')
		label_day.pack()
	else:
		label_day = Label(right_frame, text="it's a runny day",font=("Arial",14),bg='#619ecb')
		label_day.pack()
	year_input.delete(0, tk.END)
	month_input.delete(0, tk.END)
	day_input.delete(0, tk.END)

# creer la fenetre
window = Tk()
#personnaliser la fenetre
window.title("Weather prediction")
window.config(background='#619ecb')
window.geometry("1024x1024")
frame = Frame(window, bg='#619ecb')

#creer un titre
label_title = Label(window, text="Weather Predictin ",font=("Arial",40),bg='#619ecb')
label_title.pack(pady=(30,0))

#creation de l'image
width = 1024
height = 200
image = PhotoImage(file="back.png")
canvas = Canvas(window,width=width,height=height,bg='#619ecb', bd=0, highlightthickness = 0)
canvas.create_image(width/2,height/2,image=image)
#canvas.grid(row=0,column=0,sticky=W,padx=(10,20))
canvas.pack()
#creer une sous boite
right_frame = Frame(frame, bg='#619ecb')

#creer un titre
label_year = Label(right_frame, text="L'année de prédiction",font=("Arial",14),bg='#619ecb')
label_year.pack()

#creer une entrée/input
year_input = Entry(right_frame,font=("Arial",20),bg='#619ecb')
year_input.pack(pady=(0,20))

#creer un titre
label_month = Label(right_frame, text="Le mois de prédiction",font=("Arial",14),bg='#619ecb')
label_month.pack()

#creer une entrée/input
month_input = Entry(right_frame,font=("Arial",20),bg='#619ecb')
month_input.pack(pady=(0,20))

#creer un titre
label_day = Label(right_frame, text="Le jour de prédiction",font=("Arial",14),bg='#619ecb')
label_day.pack()

#creer une entrée/input
day_input = Entry(right_frame,font=("Arial",20),bg='#619ecb')
day_input.pack(pady=(0,20))

#ajouter un boutons
button = Button(right_frame, text="predict",font=("Arial",14), bg='#0a1f2f', bd=0, relief=SUNKEN, command=show_entry_fields)
button.pack(pady=(10,0))

#placer le sous frame a droite du frame principle
right_frame.grid(row=0,column=1,sticky=W)



#centrer le frame
frame.pack(expand=YES)
# affichage
window.mainloop()

# creer la frame principale
#frame = Frame(window, bg='#dee5dc')
#frame2 = Frame(frame, bg='#dee5dc')
#frame3 = Frame(frame2, bg='#dee5dc')
#frame4 = Frame(frame3, bg='#dee5dc')
# ajout du bouton/image
#button = Button(frame, text="predict", bg='#dee5dc', bd=0, relief=SUNKEN, command=add_cookie)
#button.pack()

#year
#name = tk.Label(frame3, text="Year")
#name.pack(side="left")
#e1 = tk.Entry(frame3)
#e1.pack(side="left")
#month
#surname = tk.Label(frame4, text="Month")
#surname.pack(side="left")
#e2 = tk.Entry(frame4)
#e2.pack(side="left")
#day
#day = tk.Label(frame2, text="Year")
#day.pack(side="left")
#e3 = tk.Entry(frame2)
#e3.pack(side="left")



# ajout de la frame au centre
#frame4.pack()
#frame2.pack()
#frame3.pack()






