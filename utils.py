# ==========================================
# IMPORT LIBRARIES
# ==========================================

import joblib
import pandas as pd

# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset(file_path):
    """
    Load CSV dataset.
    """
    return pd.read_csv(file_path)

# ==========================================
# SAVE OBJECT (MODEL / ENCODER)
# ==========================================

def save_object(obj, file_path):
    """
    Save model or encoder.
    """
    joblib.dump(obj, file_path)

# ==========================================
# LOAD OBJECT (MODEL / ENCODER)
# ==========================================

def load_object(file_path):
    """
    Load saved model or encoder.
    """
    return joblib.load(file_path)

# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

def clean_column_names(df):
    """
    Remove extra spaces from column names.
    """
    df.columns = df.columns.str.strip()
    return df

# ==========================================
# REMOVE DUPLICATES
# ==========================================

def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates()

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

def fill_missing_values(df):
    """
    Fill missing values:
    Numeric -> Median
    Categorical -> Mode
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

def dataset_info(df):
    """
    Print dataset information.
    """
    print("\nDataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst Five Rows:")
    print(df.head())