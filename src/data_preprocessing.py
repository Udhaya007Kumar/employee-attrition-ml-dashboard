import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =====================================
# BASE DIRECTORY
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================
# FILE PATHS
# =====================================

raw_data_path = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "employee_attrition_dataset.csv"
)

processed_dir = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

processed_data_path = os.path.join(
    processed_dir,
    "clean_data.csv"
)

# =====================================
# CREATE PROCESSED FOLDER
# =====================================

os.makedirs(
    processed_dir,
    exist_ok=True
)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv(raw_data_path)

print("\nDataset Loaded Successfully\n")

# =====================================
# DATASET INFO
# =====================================

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================
# DROP UNNECESSARY COLUMNS
# =====================================

drop_cols = [
    'EmployeeNumber',
    'Employee_ID'
]

for col in drop_cols:

    if col in df.columns:

        df.drop(
            col,
            axis=1,
            inplace=True
        )

# =====================================
# HANDLE MISSING VALUES
# =====================================

numeric_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns

for col in numeric_cols:

    df[col] = df[col].fillna(
        df[col].median()
    )

categorical_cols = df.select_dtypes(
    include='object'
).columns

for col in categorical_cols:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )

print("\nMissing Values Handled")

# =====================================
# LABEL ENCODING
# =====================================

le = LabelEncoder()

categorical_cols = df.select_dtypes(
    include='object'
).columns

for col in categorical_cols:

    df[col] = le.fit_transform(
        df[col]
    )

print("\nEncoding Completed")

# =====================================
# TARGET DISTRIBUTION
# =====================================

print("\nAttrition Distribution:\n")

print(df['Attrition'].value_counts())

# =====================================
# SAVE CLEAN DATA
# =====================================

df.to_csv(
    processed_data_path,
    index=False
)

print("\nClean Data Saved Successfully")

print("\nSaved Path:")
print(processed_data_path)

print("\nFinal Shape:")
print(df.shape)