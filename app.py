from flask import Flask, render_template, jsonify, request
from knn import Knn
import pandas as pd
import json

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/knn", methods=["POST"])
def knn():
    data = json.loads(request.data)

    p = data.get("pregnancies")
    g = data.get("glucose")
    b = data.get("bloodPressure")
    s = data.get("skinThickness")
    i = data.get("insulin")
    bmi = data.get("bmi")
    d = data.get("diabetesPedigreeFunction")
    a = data.get("age")

    # lst = [['2', '122', '70', '27', '0', '36.8', '0.34', '27']]
    df = pd.DataFrame([[p, g, b, s, i, bmi, d, a]],
                      columns=[
                          'Pregnancies', 'Glucose', 'BloodPressure',
                          'SkinThickness', 'Insulin', 'BMI',
                          'DiabetesPedigreeFunction', 'Age'
                      ])

    return jsonify(Knn(df).knn())


if __name__ == "__main__":
    app.run(debug=True)
