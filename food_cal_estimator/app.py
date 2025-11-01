import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from config import get_db_connection
import os
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "vaibhavi_secret_key"  # secure key for session handling

# Folder for uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Helper: Clear uploads folder on startup ---
def clear_uploads():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_uploads()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('home.html', title="Home")

# SIGNUP
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # check if email already exists
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()
            if existing_user:
                flash("Email already registered. Please log in instead.")
                return redirect(url_for('login'))

            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, password)
            )
            conn.commit()
            flash("Signup successful! Please log in.")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return render_template('signup.html', title="Signup")

# LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f"Welcome, {user['username']}!")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password.")

    return render_template('login.html', title="Login")

# DASHBOARD
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'], title="Dashboard")
    else:
        flash("Please log in first.")
        return redirect(url_for('login'))

from difflib import get_close_matches

@app.route('/predict_text', methods=['POST'])
def predict_text():
    if 'username' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    food_item = request.form['food_item'].strip().lower()
    quantity = request.form.get('quantity', '100').strip()  # default 100g if not provided

    # Load your dataset
    df = pd.read_csv("dataset/food_nutrition.csv")

    # 🔹 fix: actual column name is 'name', not 'food_name'
    df['name'] = df['name'].astype(str).str.lower()

    # Try exact match first
    if food_item in df['name'].values:
        calories_per_100g = df.loc[df['name'] == food_item, 'calories'].values[0]
        matched_food = food_item
    else:
        # Fuzzy match (close spelling)
        matches = get_close_matches(food_item, df['name'], n=1, cutoff=0.6)
        if matches:
            matched_food = matches[0]
            calories_per_100g = df.loc[df['name'] == matched_food, 'calories'].values[0]
        else:
            matched_food = "Not found"
            calories_per_100g = None

    # Calculate total calories
    if calories_per_100g is not None:
        try:
            quantity = float(quantity)
            total_calories = round((calories_per_100g * quantity) / 100, 2)
        except ValueError:
            total_calories = "Invalid quantity"
    else:
        total_calories = "Not found in dataset"

    return render_template(
        'dashboard.html',
        username=session['username'],
        predicted_item=matched_food,
        predicted_value=total_calories,
        title="Dashboard"
    )



# LOGOUT
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
