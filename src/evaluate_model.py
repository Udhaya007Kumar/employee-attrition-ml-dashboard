import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
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

model_path = os.path.join(
    BASE_DIR,
    "models",
    "decision_tree_model.pkl"
)

reports_dir = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    reports_dir,
    exist_ok=True
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(clean_data_path)

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

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
# LOAD MODEL
# =====================================

model = joblib.load(model_path)

# =====================================
# PREDICTIONS
# =====================================

y_pred = model.predict(X_test)

# =====================================
# METRICS
# =====================================

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(precision_score(y_test, y_pred))

print("\nRecall:")
print(recall_score(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        reports_dir,
        "confusion_matrix.png"
    )
)

plt.show()

# =====================================
# ROC CURVE
# =====================================

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(6,4))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.2f}"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    os.path.join(
        reports_dir,
        "roc_curve.png"
    )
)

plt.show()

# =====================================
# FEATURE IMPORTANCE
# =====================================

importance = model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_df
)

plt.title("Feature Importance")

plt.savefig(
    os.path.join(
        reports_dir,
        "feature_importance.png"
    )
)

plt.show()