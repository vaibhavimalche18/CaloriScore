import os
import pandas as pd

# ✅ Correct paths
CSV_PATH = r"C:\Users\vaibh\OneDrive\Desktop\food_cal_estimator\dataset\Indian_Food_Nutrition_Processed.csv"
IMAGE_BASE_PATH = r"C:\Users\vaibh\OneDrive\Desktop\food_cal_estimator\dataset\Indian Food Images\Indian Food Images"

# ✅ Load CSV
df = pd.read_csv(CSV_PATH)
df["Dish Name"] = df["Dish Name"].str.strip().str.lower()

csv_dishes = set(df["Dish Name"].tolist())
print("\n🍽 Total dishes in CSV:", len(csv_dishes))

# ✅ Verify image path exists
print("\n📌 Checking image path...")
print("Path:", IMAGE_BASE_PATH)
print("Exists:", os.path.exists(IMAGE_BASE_PATH))

# ✅ List folders inside image dataset
folders = os.listdir(IMAGE_BASE_PATH)
print("\n📁 Total folders in images:", len(folders))
print("\n🔎 Checking matches...\n")

matches = []

for folder in folders:
    folder_clean = folder.lower().replace("_", " ").strip()

    print(f"Folder: {folder:30}  → Cleaned: {folder_clean:30}  → Match: {folder_clean in csv_dishes}")

    if folder_clean in csv_dishes:
        matches.append(folder_clean)

print("\n✅ Total matching dish names found:", len(matches))
print("✅ Matching dishes:", matches)
