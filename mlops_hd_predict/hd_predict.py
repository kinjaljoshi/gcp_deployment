import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from flask import Flask, render_template, request
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
    gender = request.form.get('gender')
    height = request.form.get('height')
    weight = int(request.form['weight'])
    bphigh = int(request.form['bphigh'])
    bplow = request.form.get('bplow')
    cholestrol = int(request.form['cholesterol'])
    glucose = int(request.form['glucose'])
    smoking = request.form.get('smoking')
    alcohol = float(request.form['alcohol'])
    exercise = float(request.form['exercise'])
        
    data = np.array([[age,gender,height,weight,bphigh,bplow,cholestrol,glucose,smoking,alcohol,exercise]])
	
    # Load the object from the .pkl file
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)

    hd_predict = model.predict(data)

    return render_template('result.html', prediction=hd_predict)     

if __name__ == '__main__':
	app.run(debug=True)



