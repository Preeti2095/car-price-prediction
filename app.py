from flask import Flask, render_template, request
# import jsonify
# import requests
import pickle
# import numpy as np

from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open("random_forest_regressor.pkl", "rb"))


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")
    # return render_template("hello")


standard_to = StandardScaler()


@app.route("/predict", methods=["POST"])
def predict():
    # fuel_type_diesel = 0
    if request.method == 'POST':
        year = int(request.form['Year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])

        owner = int(request.form['owner'])
        fuel_type_petrol = request.form['fuel_type_petrol']
        if fuel_type_petrol == 'Petrol':
            fuel_type_petrol = 1
            fuel_type_diesel = 0
        elif fuel_type_petrol == 'Diesel':
            fuel_type_petrol = 0
            fuel_type_diesel = 1
        else:
            fuel_type_petrol = 0
            fuel_type_diesel = 0
        year = 2021-year
        seller_type_individual = request.form['seller_type_individual']
        if seller_type_individual == 'individual':
            seller_type_individual = 1
        else:
            seller_type_individual = 0

        transmission_manual = request.form['transmision_manual']
        if transmission_manual == 'Mannual':
            transmission_manual = 1
        else:
            transmission_manual = 0
        prediction = model.predict([[present_price, kms_driven, owner, year, fuel_type_diesel, fuel_type_petrol, seller_type_individual, transmission_manual]])
        # print(prediction)
        output = round(prediction[0], 2)
        # print(output)
        if output < 0:

            return render_template('index.html', prediction_texts="sorry you cannot sell this car")
        else:

            return render_template('index.html', prediction_texts="you can sell this car at {} lakhs".format(output))

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
