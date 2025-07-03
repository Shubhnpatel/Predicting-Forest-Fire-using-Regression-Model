import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from flask import Flask,request, jsonify,render_template


application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickel 
ridgeModel = pickle.load(open('Models/ridge.pkl','rb'))
standardScaler = pickle.load(open('Models/scaler.pkl','rb'))
print(ridgeModel.coef_.shape)


## Home Page 

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predictDatapoint():
    if request.method == "POST":
        Tempreature = float(request.form.get('Tempreature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        newScaledData = standardScaler.transform([[Tempreature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridgeModel.predict(newScaledData)

        return render_template('home.html',results = result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")