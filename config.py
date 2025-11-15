import mysql.connector

def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",          # 🔹 Replace with your MySQL username
        password="Bhagya@27",  # 🔹 Replace with your MySQL password
        database="food_estimator"
    )
    return conn
