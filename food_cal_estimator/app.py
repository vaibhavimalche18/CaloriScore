from flask import Flask, render_template, request, redirect, url_for, session, flash
from config import get_db_connection
import os

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

# LOGOUT
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
