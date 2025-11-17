import os
import json
import cv2
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from difflib import get_close_matches

app = Flask(__name__)
app.secret_key = "vaibhavi_secret_key"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER   # ✅ THIS WAS MISSING
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "food_model_best.h5")
CSV_PATH = os.path.join(BASE_DIR, "model", "food_calories.csv")


# Load model
CNN_MODEL = load_model("model/food_cnn_model.h5")
print("✅ CNN model loaded successfully.")

# Load class index mapping
with open("model/class_indices.json", "r") as f:
    INDEX_TO_CLASS = json.load(f)
print("✅ Class index mapping loaded.")




# ======================================================
# LOAD CLASS INDICES (folder → class index mapping)
# ======================================================
try:
    with open("model/class_indices.json", "r") as file:
        class_indices = json.load(file)

# Reverse mapping: index → class name
    INDEX_TO_CLASS = {int(v): k for k, v in class_indices.items()}
    print("INDEX_TO_CLASS:", INDEX_TO_CLASS)  # Debug check
    print("✅ Class index mapping loaded.")
except Exception as e:
    print("❌ Error loading class_indices.json:", e)
    idx_to_class = {}


# ======================================================
# LOAD NUTRITION CSV (food_item, calories, contents)
# ======================================================
    CSV_PATH = os.path.join(BASE_DIR, "model", "food_calories.csv")

try:
    nutrition_df = pd.read_csv(CSV_PATH)
    nutrition_df["food_item"] = nutrition_df["food_item"].astype(str).str.lower()
    print("✅ Nutrition CSV loaded.")
except Exception as e:
    print(f"⚠️ Error loading food_calories.csv: {e}")
    nutrition_df = pd.DataFrame()

    nutrition_df["food_item"] = nutrition_df["food_item"].astype(str).str.lower()
    print("✅ Nutrition CSV loaded.")
except Exception as e:
    print("⚠️ Error loading food_calories.csv:", e)
    nutrition_df = None


# Load meal planner dataset once when app starts
MEAL_DF = pd.read_csv("dataset/meal_planner_dataset.csv")

# Normalize column names (just in case)
MEAL_DF.columns = [c.strip().lower() for c in MEAL_DF.columns]
# Expecting columns: food_name, calories, category, diet_type
# If diet_type missing, create simple mapping from category
if 'diet_type' not in MEAL_DF.columns:
    MEAL_DF['diet_type'] = MEAL_DF.get('category', 'veg').fillna('veg').apply(lambda x: x.lower())
else:
    MEAL_DF['diet_type'] = MEAL_DF['diet_type'].fillna('veg').str.lower()

# Course/meal time: if dataset has a 'category' column that designates breakfast/lunch/dinner,
# otherwise we will not filter by meal_time (we'll use whole dataset). If you have a 'course' column, map it.
if 'course' in MEAL_DF.columns:
    MEAL_DF['course'] = MEAL_DF['course'].fillna('').str.lower()
else:
    MEAL_DF['course'] = ''  # blank means unknown
    



def predict_food_from_image(filepath):
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = CNN_MODEL.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    
    print("Prediction probabilities:", prediction)
    predicted_index = int(np.argmax(prediction))
    predicted_food = INDEX_TO_CLASS.get(str(predicted_index), "Unknown")

    print("Predicted index:", predicted_index)
    print("Predicted food:", predicted_food)



    
    return INDEX_TO_CLASS.get(predicted_index, "Unknown")

# ======================================================
# HELPER : Delete Previous Uploaded Images
# ======================================================
def clear_uploads():
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))



def get_daily_targets(total_calories):
    breakfast_target = round(total_calories * 0.25)
    lunch_target = round(total_calories * 0.35)
    dinner_target = round(total_calories * 0.40)
    return breakfast_target, lunch_target, dinner_target

def get_meal_suggestion(df, target, top_n=5):
    # safe copy, ensure calories numeric
    df_local = df.copy()
    df_local['calories'] = pd.to_numeric(df_local['calories'], errors='coerce').fillna(0)
    df_local['diff'] = (df_local['calories'] - target).abs()
    return df_local.sort_values('diff').head(top_n)[['food_name', 'calories']].to_dict(orient='records')

def recommend_meals_for_user(username, top_n=6):
    # fetch user profile
    cur = mysql.connection.cursor(dictionary=True)
    cur.execute("SELECT * FROM user_profile WHERE username=%s", (username,))
    user = cur.fetchone()
    if not user:
        return []

    diet = user.get('diet_preference') or ""
    allergies = (user.get('allergies') or "").lower().split(',')
    dislikes = (user.get('disliked_foods') or "").lower().split(',')
    goal_cal = user.get('calorie_goal') or 2000
    per_meal_target = float(goal_cal) / 3.0

    # simple SQL to fetch candidate meals (avoid complex text matching here)
    cur.execute("SELECT * FROM meals")
    meals = cur.fetchall()
    recs = []
    for m in meals:
        tags = (m.get('tags') or "").lower()
        # diet filter
        if diet.lower().startswith("veg") and not m.get('is_veg'):
            continue
        # allergies / dislikes filter (very simple substring check)
        skip = False
        for a in allergies:
            if a.strip() and a.strip() in tags:
                skip = True; break
        for d in dislikes:
            if d.strip() and d.strip() in tags:
                skip = True; break
        if skip:
            continue
        # score closeness to per-meal calories (smaller difference => higher score)
        score = -abs((m['calories_per_serving'] or 0) - per_meal_target)
        # optionally use popularity to boost
        score += (m.get('popularity',0) * 0.01)
        recs.append((score, m))

    recs.sort(reverse=True, key=lambda x: x[0])
    return [r[1] for r in recs[:top_n]]


# ======================================================
# ROUTES
# ======================================================
@app.route('/')
def home():
    return render_template("home.html", title="Home")

@app.route("/signup")
def signup():
    return render_template("signup.html", title="Signup")



# @app.route('/login', methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         username = request.form["username"]
#         session["username"] = username
#         flash(f"Welcome {username}!")
#         return redirect(url_for("dashboard"))
#     return render_template("login.html", title="Login")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check user exists
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s",
                       (username, password))
        user = cursor.fetchone()

        if user:
            session['username'] = username

            # AUTO CREATE PROFILE IF NOT EXISTS
            cursor.execute("SELECT * FROM user_profile WHERE username=%s", (username,))
            profile = cursor.fetchone()

            if not profile:
                cursor.execute(
                    "INSERT INTO user_profile (username) VALUES (%s)",
                    (username,)
                )
                conn.commit()

            conn.close()
            return redirect(url_for("dashboard"))

        conn.close()
        flash("Invalid credentials")
        return redirect(url_for("login"))

    return render_template("login.html")



# @app.route('/dashboard')
# def dashboard():
#     if "username" not in session:
#         flash("Please login first")
#         return redirect(url_for("login"))
#     return render_template("dashboard.html", username=session["username"], title="Dashboard")

import mysql.connector

@app.route('/dashboard')
def dashboard():
    if "username" not in session:
        return redirect('/login')

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Bhagya@27",
        database="food_estimator"
    )

    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM user_profile WHERE username=%s", (session['username'],))
    profile = cur.fetchone()

    cur.close()
    conn.close()

    return render_template("dashboard.html", profile=profile)




# ======================================================
# 🔹 TEXT SEARCH - Based on CSV
# ======================================================
@app.route('/predict_text', methods=["POST"])
def predict_text():
    if "username" not in session:
        flash("Please log in first.")
        return redirect(url_for("login"))

    food_name = request.form["food_item"].strip().lower()
    quantity = float(request.form["quantity"])

    # find exact match
    row = nutrition_df[nutrition_df["food_item"] == food_name]

    if row.empty:
        
        possible_match = get_close_matches(food_name, nutrition_df["food_item"].tolist(), n=1)

        if possible_match:
            food_name = possible_match[0]
            row = nutrition_df[nutrition_df["food_item"] == food_name]
        else:
            return render_template(
                "dashboard.html",
                username=session["username"],
                predicted_item="Not Found",
                predicted_value="N/A",
                title="Dashboard",
            )

    cal_100g = row["calories"].values[0]
    contents = row["contents"].values[0]

    total_cal = round((cal_100g * quantity) / 100, 2)

    return render_template(
        "dashboard.html",
        username=session["username"],
        predicted_item=food_name.capitalize(),
        predicted_value=total_cal,
        ingredients=contents,
        title="Dashboard"
    )


# ======================================================
# 🔹 CNN IMAGE ANALYSIS + CALORIE / INGREDIENT FETCHING
# ======================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "username" not in session:
        flash("Please log in first.")
        return redirect(url_for("login"))

    if "food_image" not in request.files:
        flash("Upload an image first.")
        return redirect(url_for("dashboard"))

    file = request.files["food_image"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("dashboard"))

    clear_uploads()
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Predict food from image
    detected_food = predict_food_from_image(filepath)

    # Fix underscores, title case, and lookup case-insensitively
    detected_food_readable = detected_food.replace("_", " ").title()
    row = nutrition_df[nutrition_df["food_item"].str.lower() == detected_food_readable.lower()]

    if not row.empty:
        calories = row["calories"].values[0]
        ingredients = row["contents"].values[0]
    else:
        calories = "Not available"
        ingredients = "N/A"

    return render_template(
        "dashboard.html",
        username=session["username"],
        image_filename=file.filename,
        food_name=detected_food_readable,
        calories=calories,
        ingredients=ingredients,
        title="Dashboard"
    )


@app.route("/meal_plan")
def meal_plan():

    diet = request.args.get("diet")
    goal = int(request.args.get("calories"))

    df = pd.read_csv("dataset/meal_planner_dataset.csv")

    # Normalize CSV entries
    df["diet_type"] = df["diet_type"].str.lower().str.strip()
    df["diet_type"] = df["diet_type"].replace({
        "non veg": "non-veg",
        "nonveg": "non-veg",
        "veg": "veg",
        "glutan free": "gluten-free",
        "gluten free": "gluten-free",
        "egg": "eggetarian",
        "eggetarian": "eggetarian"
    })

    # Normalize user input
    diet = diet.lower().strip().replace(" ", "-").replace("_", "-")

    diet_alias = {
        "veg": "veg",
        "vegetarian": "veg",

        "nonveg": "non-veg",
        "non-veg": "non-veg",
        "non-vegetarian": "non-veg",

        "glutenfree": "gluten-free",
        "gluten-free": "gluten-free",

        "egg": "eggetarian",
        "eggetarian": "eggetarian",

        "vegan": "vegan"
    }

    diet = diet_alias.get(diet, diet)

    # -------------------------
    # Meal Filtering
    # -------------------------
    if diet == "veg":
        df = df[df["diet_type"] == "veg"]

    elif diet == "non-veg":
        df = df[df["diet_type"].isin(["veg", "non-veg"])]

    elif diet == "gluten-free":
        df = df[df["diet_type"] == "gluten-free"]

    elif diet == "vegan":
        df = df[df["diet_type"] == "vegan"]

    elif diet == "eggetarian":
        df = df[df["diet_type"] == "eggetarian"]

    else:
        return f"Unknown diet: {diet}"

    if df.empty:
        return f"No meals found for diet: {diet}"

    # -------------------------
    # UPDATED NON-VEG LOGIC
    # -------------------------
    def select_meals(meal_type, diet):
        meal_df = df[df["meal_type"].str.contains(meal_type, case=False, na=False)]
        if meal_df.empty:
            return [], 0

        veg_items = meal_df[meal_df["diet_type"] == "veg"]
        nonveg_items = meal_df[meal_df["diet_type"] == "non-veg"]

        selected = []
        total_cal = 0
        limit = goal / 3

        # Force at least 1 non-veg for non-veg diet
        if diet == "non-veg" and not nonveg_items.empty:
            for _, row in nonveg_items.iterrows():
                if total_cal + row["calories"] <= limit:
                    selected.append(row)
                    total_cal += row["calories"]
                    break

        # Fill remaining calories with veg + non-veg
        remaining = pd.concat([veg_items, nonveg_items]).sort_values("calories")

        for _, row in remaining.iterrows():
            if total_cal + row["calories"] <= limit:
                selected.append(row)
                total_cal += row["calories"]

        return selected, total_cal

    # -------------------------
    # Generate Meals
    # -------------------------
    breakfast_items, breakfast_total = select_meals("breakfast", diet)
    lunch_items, lunch_total = select_meals("lunch", diet)
    dinner_items, dinner_total = select_meals("dinner", diet)

    return render_template(
        "meal_result.html",
        diet=diet.capitalize(),
        breakfast_items=breakfast_items,
        lunch_items=lunch_items,
        dinner_items=dinner_items,
        breakfast_total=breakfast_total,
        lunch_total=lunch_total,
        dinner_total=dinner_total,
        day_total=breakfast_total + lunch_total + dinner_total,
        goal=goal
    )

####################################################################COMPUTE THE BMI
def compute_bmi(weight_kg, height_cm):
    if not weight_kg or not height_cm:
        return None
    h = float(height_cm) / 100.0
    try:
        bmi = float(weight_kg) / (h * h)
        return round(bmi, 2)
    except Exception:
        return None

def compute_bmr(weight, height, age, gender):
    # expects weight in kg, height in cm, age in years
    if gender.lower() == "male":
        return 88.362 + (13.397 * float(weight)) + (4.799 * float(height)) - (5.677 * float(age))
    else:
        return 447.593 + (9.247 * float(weight)) + (3.098 * float(height)) - (4.330 * float(age))

ACTIVITY_MULT = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725
}

def compute_daily_calorie_goal(weight, height, age, gender, activity_level, goal):
    try:
        bmr = compute_bmr(weight, height, age, gender)
        mult = ACTIVITY_MULT.get(activity_level, 1.2)
        maintenance = bmr * mult
        if goal.lower().startswith("weight loss"):
            return round(max(1200, maintenance - 500), 2)   # floor at 1200
        elif goal.lower().startswith("weight gain"):
            return round(maintenance + 300, 2)
        else:
            return round(maintenance, 2)
    except Exception:
        return None


#############add the db details:
import mysql.connector

def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Bhagya@27",
        database="food_estimator"
    )
    return conn



@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    return render_template("profile.html", username=username)

#################################################################################---------SAVE PROFILE
import mysql.connector

@app.route('/save_profile', methods=['POST'])
def save_profile():

    username = request.form["username"]
    age = int(request.form["age"])
    gender = request.form["gender"]
    height = float(request.form["height"])
    weight = float(request.form["weight"])
    activity = request.form["activity"]
    goal = request.form["goal"]
    diet_pref = request.form["diet_pref"]
    allergies = request.form["allergies"]
    dislikes = request.form["dislikes"]

    # BMI calculation
    height_m = height / 100
    bmi = weight / (height_m * height_m)

    # Daily calories
    if gender == "Male":
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

    activity_factors = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }
    maintenance_calories = bmr * activity_factors[activity]

    if goal == "Weight Loss":
        daily_calories = maintenance_calories - 500
    elif goal == "Weight Gain":
        daily_calories = maintenance_calories + 300
    else:
        daily_calories = maintenance_calories

    # CONNECT USING mysql.connector
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Bhagya@27",
        database="food_estimator"
    )

    cursor = conn.cursor()

    query = """
        UPDATE user_profile SET 
            age=%s, gender=%s, height=%s, weight=%s,
            activity_level=%s, goal=%s, diet_preference=%s,
            allergies=%s, disliked_foods=%s, bmi=%s, daily_calorie_need=%s
        WHERE username=%s
    """

    values = (
        age, gender, height, weight,
        activity, goal, diet_pref,
        allergies, dislikes, bmi, daily_calories,
        username
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()

    return redirect('/dashboard')








@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)