import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ Paths (Keep these correct)
CSV_PATH = r"C:\Users\vaibh\OneDrive\Desktop\food_cal_estimator\dataset\Indian_Food_Nutrition_Processed.csv"
IMAGE_BASE_PATH = r"C:\Users\vaibh\OneDrive\Desktop\food_cal_estimator\dataset\Indian Food Images\Indian Food Images"
OUTPUT_PATH = "dataset"

TRAIN_PATH = os.path.join(OUTPUT_PATH, "train")
VAL_PATH = os.path.join(OUTPUT_PATH, "validation")

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VAL_PATH, exist_ok=True)

# ✅ Load CSV
df = pd.read_csv(CSV_PATH)

# Normalize dish names: lowercase + remove extra spaces
df["Dish Name"] = df["Dish Name"].str.strip().str.lower()

# ✅ Create a lookup dictionary: dish → calories
dish_calorie_map = dict(zip(df["Dish Name"], df["Calories (kcal)"]))

# ✅ Iterate image folders to find matching dish names
dataset_rows = []

for folder in os.listdir(IMAGE_BASE_PATH):
    folder_clean = folder.lower().replace("_", " ").strip()  # aloo_gobi -> aloo gobi

    if folder_clean in dish_calorie_map.keys():
        dataset_rows.append({
            "folder": folder,
            "dish_clean": folder_clean,
            "calories": dish_calorie_map[folder_clean]
        })

if not dataset_rows:
    print("❌ No matching dish names found. Check naming.")
    exit()

dataset_df = pd.DataFrame(dataset_rows)

# ✅ Train / validation split (80/20)
train_df, val_df = train_test_split(dataset_df, test_size=0.2, random_state=42)

def copy_images(df_rows, out_folder):
    for _, row in df_rows.iterrows():
        src = os.path.join(IMAGE_BASE_PATH, row["folder"])
        dst = os.path.join(out_folder, row["folder"])
        shutil.copytree(src, dst, dirs_exist_ok=True)

print("Copying training images...")
copy_images(train_df, TRAIN_PATH)

print("Copying validation images...")
copy_images(val_df, VAL_PATH)

print("\n✅ Dataset ready!")
print("📁 Training images saved at:", TRAIN_PATH)
print("📁 Validation images saved at:", VAL_PATH)
