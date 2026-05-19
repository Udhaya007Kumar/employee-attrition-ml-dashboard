import joblib
import pandas as pd

model = joblib.load("models/decision_tree_model.pkl")

def predict_employee(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return prediction[0]