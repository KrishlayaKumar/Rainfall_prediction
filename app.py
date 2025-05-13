import flask
from flask import Flask, render_template, request
from flask_cors import cross_origin 
from tensorflow.keras.models import load_model 
import pickle
import numpy as np

app = Flask(__name__, template_folder="template")

# Load Keras model and scaler from the specified location
model = load_model("C:/Users/ASUS/Desktop/MINI_project/model/ANN/ann_model.h5")
scaler = pickle.load(open("C:/Users/ASUS/Desktop/MINI_project/model/ANN/scaler.pkl", "rb"))

print("Model Loaded")

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        month = float(request.form['date'])
        minTemp = float(request.form['mintemp'])
        maxTemp = float(request.form['maxtemp'])
        rainfall = float(request.form['rainfall'])
        evaporation = float(request.form['evaporation'])
        sunshine = float(request.form['sunshine'])
        windGustSpeed = float(request.form['windgustspeed'])
        windSpeed9am = float(request.form['windspeed9am'])
        windSpeed3pm = float(request.form['windspeed3pm'])
        humidity9am = float(request.form['humidity9am'])
        humidity3pm = float(request.form['humidity3pm'])
        pressure9am = float(request.form['pressure9am'])
        pressure3pm = float(request.form['pressure3pm'])
        temp9am = float(request.form['temp9am'])
        temp3pm = float(request.form['temp3pm'])
        cloud9am = float(request.form['cloud9am'])
        cloud3pm = float(request.form['cloud3pm'])
        location = float(request.form['location'])
        winddDir9am = float(request.form['winddir9am'])
        winddDir3pm = float(request.form['winddir3pm'])
        windGustDir = float(request.form['windgustdir'])
        rainToday = float(request.form['raintoday'])

        input_lst = [[location, minTemp, maxTemp, rainfall, evaporation, sunshine,
                      windGustDir, windGustSpeed, winddDir9am, winddDir3pm, windSpeed9am, windSpeed3pm,
                      humidity9am, humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm,
                      temp9am, temp3pm, rainToday, month]]

        input_data = scaler.transform(input_lst)
        output_prob = model.predict(input_data)
        output = int(output_prob[0][0] >= 0.5)

        if output == 0:
            return render_template("sunny.html")
        else:
            return render_template("rainy.html")

if __name__ == '__main__':
    app.run(debug=True)
