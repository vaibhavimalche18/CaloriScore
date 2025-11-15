# utils/create_food_calories.py
import pandas as pd
nut = pd.read_csv("C:\Users\vaibh\OneDrive\Desktop\food_cal_estimator\dataset\food_nutrition.csv")  # or path to your tabular nutrition CSV
# assume 'name' column and 'calories' column exist
nut['name_norm'] = nut['name'].str.strip().str.lower()
# pick one row per food_name - customizing may be needed
mapping = nut.groupby('name_norm').agg({
    'name': 'first',
    'calories': 'first'
}).reset_index()
mapping['contents'] = mapping['name']  # placeholder, update manually if needed
mapping[['name','calories','contents']].to_csv("model/food_calories.csv", index=False,
                                              header=['food_item','calories','contents'])
print("✅ auto-created model/food_calories.csv (review and edit contents column)")
