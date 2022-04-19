from flask import Flask, render_template, request
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

    try:
        p = float(data.get("pregnancies"))
        g = float(data.get("glucose"))
        b = float(data.get("bloodPressure"))
        s = float(data.get("skinThickness"))
        i = float(data.get("insulin"))
        bmi = float(data.get("bmi"))
        d = float(data.get("diabetesPedigreeFunction"))
        a = float(data.get("age"))
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)

    # lst = [['2', '122', '70', '27', '0', '36.8', '0.34', '27']]
    df = pd.DataFrame([[p, g, b, s, i, bmi, d, a]],
                      columns=[
                          'Pregnancies', 'Glucose', 'BloodPressure',
                          'SkinThickness', 'Insulin', 'BMI',
                          'DiabetesPedigreeFunction', 'Age'
                      ])

    return Knn(df).knn()


if __name__ == "__main__":
    app.run(debug=True)
