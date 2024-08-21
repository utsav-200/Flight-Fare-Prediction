# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:52:38 2024

@author: KIIT
"""

from flask import Flask, request, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = pickle.load(open("flight_rf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").minute)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").minute)

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        # Total Stops
        Total_stops = int(request.form["stops"])

        # Airline
        airline = request.form['airline']
        airline_dict = {
            'Jet Airways': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IndiGo': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Air India': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'Multiple carriers': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'SpiceJet': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'Vistara': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'GoAir': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'Multiple carriers Premium economy': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'Jet Airways Business': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'Vistara Premium economy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'Trujet': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        airline_features = airline_dict.get(airline, [0] * 11)

        # Source
        source = request.form["Source"]
        source_dict = {
            'Delhi': [1, 0, 0, 0],
            'Kolkata': [0, 1, 0, 0],
            'Mumbai': [0, 0, 1, 0],
            'Chennai': [0, 0, 0, 1]
        }
        source_features = source_dict.get(source, [0] * 4)

        # Destination
        destination = request.form["Destination"]
        destination_dict = {
            'Cochin': [1, 0, 0, 0, 0],
            'Delhi': [0, 1, 0, 0, 0],
            'New_Delhi': [0, 0, 1, 0, 0],
            'Hyderabad': [0, 0, 0, 1, 0],
            'Kolkata': [0, 0, 0, 0, 1]
        }
        destination_features = destination_dict.get(destination, [0] * 5)

        # Feature vector
        features = [
            Total_stops, Journey_day, Journey_month,
            Dep_hour, Dep_min, Arrival_hour, Arrival_min,
            dur_hour, dur_min
        ] + airline_features + source_features + destination_features

        prediction = model.predict([features])
        output = round(prediction[0], 2)

        return render_template('home.html', prediction_text=f"Your Flight price is Rs. {output}")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
