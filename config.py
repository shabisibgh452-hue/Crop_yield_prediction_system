import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATASET_DIR, "yield_df.csv")
CLEAN_DATA_PATH = os.path.join(DATASET_DIR, "crop_cleaned.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_crop_model.pkl")

AREA_ENCODER_PATH = os.path.join(MODELS_DIR, "area_encoder.pkl")
ITEM_ENCODER_PATH = os.path.join(MODELS_DIR, "item_encoder.pkl")

TEST_SIZE = 0.20
RANDOM_STATE = 42

TARGET_COLUMN = "hg/ha_yield"