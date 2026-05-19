import os
import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(

    page_title="ML Dashboard",

    page_icon="📊",

    layout="wide"
)

# =====================================
# BASE DIRECTORY
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================
# MODEL PATHS
# =====================================

classification_model_path = os.path.join(

    BASE_DIR,

    "models",

    "decision_tree_model.pkl"
)

regression_model_path = os.path.join(

    BASE_DIR,

    "models",

    "decision_tree_regression.pkl"
)

# =====================================
# LOAD MODELS
# =====================================

classification_model = joblib.load(
    classification_model_path
)

regression_model = joblib.load(
    regression_model_path
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(

    "Choose Model",

    [
        "Classification",
        "Regression"
    ]
)

# =====================================
# CLASSIFICATION
# =====================================

if page == "Classification":

    st.title(
        "📊 Employee Attrition Prediction"
    )

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    gender_map = {
        "Male":1,
        "Female":0
    }

    gender = gender_map[gender]

    monthly_income = st.number_input(

        "Monthly Income",

        min_value=1000,

        value=5000
    )

    overtime = st.selectbox(

        "Overtime",

        ["Yes", "No"]
    )

    overtime_map = {
        "Yes":1,
        "No":0
    }

    overtime = overtime_map[overtime]

    years_company = st.slider(

        "Years At Company",

        0,

        40,

        5
    )

    if st.button(
        "Predict Attrition"
    ):

        sample = pd.DataFrame({

            'Age':[age],

            'Gender':[gender],

            'Monthly_Income':[monthly_income],

            'Overtime':[overtime],

            'Years_at_Company':[years_company]
        })

        expected_features = classification_model.feature_names_in_

        for col in expected_features:

            if col not in sample.columns:

                sample[col] = 0

        sample = sample[expected_features]

        prediction = classification_model.predict(
            sample
        )

        probability = classification_model.predict_proba(
            sample
        )

        st.divider()

        if prediction[0] == 1:

            st.error(
                "⚠️ Employee May Leave"
            )

        else:

            st.success(
                "✅ Employee Will Stay"
            )

        leave_prob = probability[0][1] * 100

        stay_prob = probability[0][0] * 100

        st.write(
            f"📉 Leave Probability: {leave_prob:.2f}%"
        )

        st.write(
            f"📈 Stay Probability: {stay_prob:.2f}%"
        )

# =====================================
# REGRESSION
# =====================================

elif page == "Regression":

    st.title(
        "💰 Salary Prediction"
    )

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    years_company = st.slider(

        "Years At Company",

        0,

        40,

        5
    )

    performance = st.slider(

        "Performance Rating",

        1,

        5,

        3
    )

    overtime = st.selectbox(

        "Overtime",

        ["Yes", "No"]
    )

    overtime_map = {
        "Yes":1,
        "No":0
    }

    overtime = overtime_map[overtime]

    if st.button(
        "Predict Salary"
    ):

        sample = pd.DataFrame({

            'Age':[age],

            'Years_at_Company':[years_company],

            'Performance_Rating':[performance],

            'Overtime':[overtime]
        })

        expected_features = regression_model.feature_names_in_

        for col in expected_features:

            if col not in sample.columns:

                sample[col] = 0

        sample = sample[expected_features]

        prediction = regression_model.predict(
            sample
        )

        st.divider()

        st.success(
            f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}"
        )