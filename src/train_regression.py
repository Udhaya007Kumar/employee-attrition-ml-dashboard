import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score
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
# PATHS
# =====================================

clean_data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clean_data.csv"
)

model_dir = os.path.join(
    BASE_DIR,
    "models"
)

model_path = os.path.join(
    model_dir,
    "decision_tree_regression.pkl"
)

os.makedirs(
    model_dir,
    exist_ok=True
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(clean_data_path)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop(
    "Monthly_Income",
    axis=1
)

y = df["Monthly_Income"]

# =====================================
# SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# =====================================
# MODEL
# =====================================

model = DecisionTreeRegressor(

    max_depth=10,

    random_state=42
)

# =====================================
# TRAIN
# =====================================

model.fit(
    X_train,
    y_train
)

# =====================================
# PREDICT
# =====================================

y_pred = model.predict(X_test)

# =====================================
# METRICS
# =====================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print("\nMAE:", mae)

print("\nMSE:", mse)

print("\nRMSE:", rmse)

print("\nR2 Score:", r2)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    model,
    model_path
)

print("\nRegression Model Saved")