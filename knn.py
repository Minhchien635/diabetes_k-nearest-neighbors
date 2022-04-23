import numpy as np
import pandas as pd
import json

import warnings

warnings.filterwarnings("ignore")
from csv import writer


class Knn:
    def __init__(self, data):
        self.input = pd.DataFrame(data)

    def knn(self):
        print(self.input)

        diabetes_data = pd.read_csv("diabetes.csv")

        for i in diabetes_data.index:
            if (
                diabetes_data.loc[i]["Pregnancies"]
                == float(self.input.loc[0]["Pregnancies"])
                and diabetes_data.loc[i]["Glucose"]
                == float(self.input.loc[0]["Glucose"])
                and diabetes_data.loc[i]["BloodPressure"]
                == float(self.input.loc[0]["BloodPressure"])
                and diabetes_data.loc[i]["SkinThickness"]
                == float(self.input.loc[0]["SkinThickness"])
                and diabetes_data.loc[i]["Insulin"]
                == float(self.input.loc[0]["Insulin"])
                and diabetes_data.loc[i]["BMI"] == float(self.input.loc[0]["BMI"])
                and diabetes_data.loc[i]["DiabetesPedigreeFunction"]
                == float(self.input.loc[0]["DiabetesPedigreeFunction"])
                and diabetes_data.loc[i]["Age"] == float(self.input.loc[0]["Age"])
            ):
                out_come = diabetes_data.loc[i]["Outcome"]
                print(out_come)
                return json.dumps(
                    {"out_come": diabetes_data.loc[i]["Outcome"]}, indent=4
                )

        with open("diabetes.csv", "a") as f_object:

            writer_object = writer(f_object)

            writer_object.writerow(list(map(float, self.input.loc[0])))

            f_object.close()

        diabetes_data = pd.read_csv("diabetes.csv")

        diabetes_data_copy = diabetes_data.copy(deep=True)
        diabetes_data_copy[
            ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        ] = diabetes_data_copy[
            ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        ].replace(
            0, np.NaN
        )

        diabetes_data_copy["Glucose"].fillna(
            diabetes_data_copy["Glucose"].mean(), inplace=True
        )
        diabetes_data_copy["BloodPressure"].fillna(
            diabetes_data_copy["BloodPressure"].mean(), inplace=True
        )
        diabetes_data_copy["SkinThickness"].fillna(
            diabetes_data_copy["SkinThickness"].median(), inplace=True
        )
        diabetes_data_copy["Insulin"].fillna(
            diabetes_data_copy["Insulin"].median(), inplace=True
        )
        diabetes_data_copy["BMI"].fillna(
            diabetes_data_copy["BMI"].median(), inplace=True
        )

        diabetes_data_copy.drop_duplicates(inplace=True, keep="last")

        from sklearn.preprocessing import StandardScaler

        sc_X = StandardScaler()
        X = pd.DataFrame(
            sc_X.fit_transform(
                diabetes_data_copy.drop(["Outcome"], axis=1),
            ),
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
        )

        last_row_scaled = X.iloc[-1:]

        X.drop(X.tail(1).index, inplace=True)

        y = diabetes_data_copy.Outcome

        y.drop(y.tail(1).index, inplace=True)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 / 3, random_state=42, stratify=y
        )

        from sklearn.neighbors import KNeighborsClassifier

        test_scores = []
        train_scores = []

        for i in range(1, 15):

            knn = KNeighborsClassifier(n_neighbors=i, metric="euclidean")
            knn.fit(X_train, y_train)

            train_scores.append(knn.score(X_train, y_train))
            test_scores.append(knn.score(X_test, y_test))

        max_test_score = max(test_scores)
        test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
        print(
            "Max test score {} % and k = {}".format(
                max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))
            )
        )

        knn = KNeighborsClassifier(n_neighbors=11, metric="euclidean")

        knn.fit(X_train, y_train)
        knn.score(X_test, y_test)

        out_come = int(knn.predict(last_row_scaled))

        # Lấy dòng cuối của file dữ liệu gán vào new_row (do dòng cuối cột Outcome chưa có giá trị)
        with open("diabetes.csv", "r+") as f_object:
            last_row = f_object.readlines()[-1]

            list_string_row = last_row.replace("\n", "").split(",")

            new_row = list(map(float, list_string_row))

            new_row.append(float(out_come))

            writer_object = writer(f_object)

            f_object.close()

        # Xóa dòng cuối file dữ liệu và ghi vào lại new_row (đã có giá trị cột Outcome)
        with open("diabetes.csv", "r+") as f:
            writer_object = writer(f)

            i = 0
            size = len(diabetes_data)

            writer_object.writerow(
                [
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age",
                    "Outcome",
                ]
            )
            while i < size - 1:
                writer_object.writerow(diabetes_data.iloc[i])
                i += 1
            writer_object.writerow(new_row)
            print(new_row)
            f.close()
            return json.dumps({"out_come": out_come}, indent=4)
