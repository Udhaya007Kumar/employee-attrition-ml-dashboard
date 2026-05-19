import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

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
    "decision_tree_model.pkl"
)

os.makedirs(
    model_dir,
    exist_ok=True
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(clean_data_path)

print(df['Attrition'].value_counts())

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# SMOTE
# =====================================

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# =====================================
# DECISION TREE
# =====================================

model = DecisionTreeClassifier(

    criterion='gini',

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight='balanced',

    random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

model.fit(
    X_train_smote,
    y_train_smote
)

# =====================================
# PREDICTION
# =====================================

y_pred = model.predict(X_test)

# =====================================
# REPORT
# =====================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    model,
    model_path
)

print("\nModel Saved Successfully")