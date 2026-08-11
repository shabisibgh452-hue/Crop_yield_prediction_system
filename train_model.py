# ==========================================
# IMPORT LIBRARIES
# ==========================================

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    CLEAN_DATA_PATH,
    BEST_MODEL_PATH,
    AREA_ENCODER_PATH,
    ITEM_ENCODER_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN
)

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

print("Loading cleaned dataset...")

df = pd.read_csv(CLEAN_DATA_PATH)

# ==========================================
# LABEL ENCODING
# ==========================================

area_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["Area"] = area_encoder.fit_transform(df["Area"])
df["Item"] = item_encoder.fit_transform(df["Item"])

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# ==========================================
# RANDOM FOREST REGRESSOR
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Accuracy (Displayed as Percentage)
accuracy = r2 * 100

print("\n====================================")
print("      MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy : {accuracy:.2f}%")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

# ==========================================
# SAVE MODEL & ENCODERS
# ==========================================

joblib.dump(model, BEST_MODEL_PATH, compress=3)
joblib.dump(area_encoder, AREA_ENCODER_PATH, compress=3)
joblib.dump(item_encoder, ITEM_ENCODER_PATH, compress=3)

print("\n====================================")
print("Model saved successfully!")
print("Area Encoder saved successfully!")
print("Item Encoder saved successfully!")
print("====================================")