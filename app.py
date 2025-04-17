from flask import Flask, request, jsonify, render_template, send_from_directory, Response, redirect, url_for
from flask_cors import CORS
import sqlite3
import os
import subprocess
import torch
import whisper
import torchaudio
from speechbrain.inference import EncoderClassifier
from deepface import DeepFace
import cv2
import time
import requests
from textblob import TextBlob
from datetime import datetime


# --- Flask Setup ---
app = Flask(__name__, template_folder="templates")
CORS(app)

# --- Database Setup ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_user_table():
    with get_db_connection() as conn:
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
def create_mood_table():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mood_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                mood TEXT NOT NULL,
                timestamp TEXT,
                date TEXT
            )
        """)
        conn.commit()
def add_polarity_column():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(mood_logs)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'polarity' not in columns:
            cursor.execute("ALTER TABLE mood_logs ADD COLUMN polarity REAL")
            conn.commit()
            print("Polarity column added.")
        else:
            print("Polarity column already exists.")


create_user_table()
create_mood_table()
add_polarity_column()


# --- ROUTES FLOW ---
@app.route("/")
def landing():
    return render_template("splash.html")  # Starting page

@app.route("/firstpage")
def first_page():
    return render_template("hm.html")

@app.route("/loginpage")
def login_page():
    return render_template("login.html")
@app.route("/registerpage")
def register_page():
    return render_template("register.html")

@app.route("/homepage")
def homepage():
    return render_template("home.html")

@app.route("/chatbot")
def chatbot():
    return render_template("index.html")  # chatbot from backend/web/index.html

@app.route("/selfcare")
def selfcare():
    return render_template("selfcare.html")

# --- Register User ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required."}), 400

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return jsonify({"error": "Email already registered."})
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, password))
            conn.commit()
        return jsonify({"message": "Registered successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Login User ---
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()

        if user:
            return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

# --- Chatbot Logic (index.html) ---
RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"

# Add this helper function above the /chat route
def get_mood(user_message, polarity):
    negative_keywords = ["sad", "anxious", "depressed", "suicidal", "stressed", "scared", "lonely", "angry", "cry", "worthless", "hopeless", "can't sleep","giving up","frustrated","don't want to live"]
    positive_keywords = ["fine", "great", "happy", "good", "awesome", "okay"]
    neutral_keywords = ["hey", "hello", "hi", "thanks", "ok", "okay", "fact", "info"]

    msg = user_message.lower()

    if any(word in msg for word in negative_keywords):
        return "negative"
    elif any(word in msg for word in positive_keywords):
        return "positive"
    elif any(word in msg for word in neutral_keywords):
        return "neutral"
    else:
        # fallback to polarity-based
        return "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    username = data.get("username", "guest")  # Add username from frontend later
    
    # Detect sentiment/mood
    blob = TextBlob(user_message)
    polarity = blob.sentiment.polarity
    mood = get_mood(user_message, polarity)
     # Debug: Print detected mood
    print(f"[DEBUG] Detected mood for '{username}': {mood} (polarity: {polarity})")

    # Get timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    date = datetime.now().strftime("%Y-%m-%d")

    # Save mood in DB
    with get_db_connection() as conn:
        cursor = conn.cursor()
          # Get the last recorded mood for this user
        cursor.execute("SELECT mood FROM mood_logs WHERE username = ? ORDER BY id DESC LIMIT 1", (username,))
        result = cursor.fetchone()
        last_mood = result[0] if result else None
        # Debug: Print last mood
        print(f"[DEBUG] Last mood for '{username}': {last_mood}")
        # Only insert if mood has changed
        if mood != last_mood:
           cursor.execute(
    "INSERT INTO mood_logs (username, mood, timestamp, date, polarity) VALUES (?, ?, ?, ?, ?)",
    (username, mood, timestamp, date, polarity)
)

        conn.commit()


    # Send message to Rasa
    rasa_response = requests.post(RASA_SERVER_URL, json={"message": user_message})
    bot_response = rasa_response.json()
    if bot_response:
        return jsonify({"response": bot_response[0]["text"]})
    else:
        return jsonify({"response": "I'm here to help. Tell me more."})


@app.route("/progress")
def progress():
    return render_template("progress.html")

@app.route("/get_mood_data", methods=["GET"])
def get_mood_data():
    username = request.args.get("username")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT mood, date, timestamp FROM mood_logs
            WHERE username = ?
            ORDER BY id ASC
        """, (username,))
        rows = cursor.fetchall()

    data = []
    mood_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for row in rows:
        mood = row["mood"]
        mood_counts[mood] += 1
        data.append({
            "mood": mood,
            "datetime": f"{row['date']} {row['timestamp']}"
        })

    return jsonify({
        "moods": data,
        "summary": mood_counts
    })
# --- Whisper & Audio ---
asr_model = whisper.load_model("base")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["file"]
    audio_file.save("temp_audio.webm")

    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", "temp_audio.webm",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "converted_audio.wav"
        ], check=True)
        signal, fs = torchaudio.load("converted_audio.wav")
        transcribed_text = asr_model.transcribe("converted_audio.wav")["text"]
        return jsonify({"transcription": transcribed_text})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

# --- Camera & Emotion ---
camera_on = False
cap = None

def generate_frames():
    global cap, camera_on
    while camera_on:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.analyze(rgb_frame, actions=["emotion"],
                                      enforce_detection=False, detector_backend="opencv")
            detected_emotion = result[0]["dominant_emotion"] if isinstance(result, list) else result["dominant_emotion"]
        except:
            detected_emotion = "neutral"

        cv2.putText(frame, f"Emotion: {detected_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    global camera_on, cap
    data = request.get_json()
    camera_on = data.get("camera_on", False)
    if camera_on:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        return jsonify({"message": "Camera ON"})
    else:
        camera_on = False
        if cap is not None:
            cap.release()
            cap = None
        return jsonify({"message": "Camera OFF"})

# --- Static files from 'backend/web' ---
@app.route("/web/<path:filename>")
def serve_web_static(filename):
    return send_from_directory("backend/web", filename)

# --- Run the App ---
if __name__ == '__main__':
    print("Using DB at:", DB_PATH)
    app.run(debug=True, port=5000)
