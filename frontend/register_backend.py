import os
from flask import Flask, request, jsonify
import sqlite3
from flask_cors import CORS
print("Using database at:", os.path.abspath("users.db"))  # 👈 This line shows the full path

app = Flask(__name__)
CORS(app)  # Allows requests from JS frontend

DB_PATH = "users.db"

def create_user_table():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

create_user_table()

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    print("Received data:", data)
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Check if email or username already exists
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                print("Email already registered!")
                return jsonify({"error": "Email already registered."})

            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, password))
            conn.commit()
            print("User inserted successfully!")
        return jsonify({"message": "Registered successfully!"})

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
