from src.preprocess import prepare_data
import os

# 1. Ensure directories exist
folders = ['data/raw', 'data/processed', 'models', 'notebooks']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 2. Define Paths
RAW_DATA_PATH = "data/raw/mental_health.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_mental_health.csv"

# 3. Execute Preprocessing
if os.path.exists(RAW_DATA_PATH):
    prepare_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
else:
    print(f"❌ Error: Please download 'mental_health.csv' and place it in {RAW_DATA_PATH}")