# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd

from config import DATASET_PATH, CLEAN_DATA_PATH
from utils import (
    load_dataset,
    clean_column_names,
    remove_duplicates,
    fill_missing_values,
    dataset_info
)

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = load_dataset(DATASET_PATH)

# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

df = clean_column_names(df)

# ==========================================
# REMOVE UNNECESSARY COLUMN
# ==========================================

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ==========================================
# CONVERT NUMERIC COLUMNS
# ==========================================

numeric_columns = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# REMOVE DUPLICATES
# ==========================================

df = remove_duplicates(df)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

df = fill_missing_values(df)

# ==========================================
# SAVE CLEANED DATASET
# ==========================================

df.to_csv(CLEAN_DATA_PATH, index=False)

# ==========================================
# DISPLAY DATASET INFO
# ==========================================

dataset_info(df)

print("\n===================================")
print("Preprocessing Completed Successfully")
print("===================================")
print(f"Cleaned dataset saved at:\n{CLEAN_DATA_PATH}")