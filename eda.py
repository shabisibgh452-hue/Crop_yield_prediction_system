# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from config import CLEAN_DATA_PATH

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv(CLEAN_DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# HISTOGRAMS
# ==========================================

numeric_columns = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
]

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=30)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# ==========================================
# CROP COUNT
# ==========================================

plt.figure(figsize=(12,5))
df["Item"].value_counts().plot(kind="bar")
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

# ==========================================
# COUNTRY COUNT
# ==========================================

plt.figure(figsize=(12,5))
df["Area"].value_counts().head(20).plot(kind="bar")
plt.title("Top 20 Countries")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()