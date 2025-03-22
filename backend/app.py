from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import requests
import torch
import whisper
from speechbrain.inference import EncoderClassifier
import torchaudio
import subprocess
import cv2
import numpy as np
from deepface import DeepFace
import threading
import time

app = Flask(__name__, static_folder="web", template_folder="web")
CORS(app)

# Load Whisper (Speech-to-Text) & SpeechBrain (Emotion Recognition)
asr_model = whisper.load_model("base")  # Whisper base model (CPU-friendly)
emotion_model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="tmp/emotion_model",
    run_opts={"local_rank": 0, "data_parallel_count": 1},
    use_auth_token=False
)

RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"

# Serve the chatbot UI
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    rasa_response = requests.post(RASA_SERVER_URL, json={"message": user_message})
    bot_response = rasa_response.json()

    if bot_response:
        return jsonify({"response": bot_response[0]["text"]})
    else:
        return jsonify({"response": "I'm here to help. Tell me more."})


# 🎤 Handle Audio Processing
@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["file"]
    original_audio_path = "temp_audio.webm"
    converted_audio_path = "converted_audio.wav"

    # Save uploaded file
    audio_file.save(original_audio_path)

    # Convert to WAV (16kHz mono) using FFmpeg
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", original_audio_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", converted_audio_path
        ], check=True)
    except subprocess.CalledProcessError:
        return jsonify({"error": "FFmpeg audio conversion failed"}), 500

    # Load and verify the converted file
    try:
        signal, fs = torchaudio.load(converted_audio_path)
    except Exception as e:
        return jsonify({"error": f"Error loading audio: {str(e)}"}), 500

    # Speech-to-Text (using Whisper)
    transcribed_text = asr_model.transcribe(converted_audio_path)["text"]

    return jsonify({"transcription": transcribed_text})


# 🎥 Webcam & Facial Emotion Recognition
camera_on = False
cap = None  # Initialize webcam capture variable


def generate_frames():
    global cap, camera_on
    while camera_on:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Reinitialize webcam if needed

        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB (for DeepFace)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect emotion
        try:
            result = DeepFace.analyze(
                rgb_frame, actions=["emotion"], enforce_detection=False, detector_backend="opencv"
            )

            if isinstance(result, list) and "dominant_emotion" in result[0]:
                detected_emotion = result[0]["dominant_emotion"]
            else:
                detected_emotion = "neutral"

        except Exception as e:
            print("Emotion detection failed:", e)
            detected_emotion = "neutral"

        print("Detected Emotion:", detected_emotion)

        # Overlay emotion text on frame
        cv2.putText(frame, f"Emotion: {detected_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        time.sleep(0.03)  # Allow some delay to prevent CPU overload


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
            cap = cv2.VideoCapture(0)  # Ensure webcam is opened properly
        return jsonify({"message": "Camera turned ON"})

    else:
        camera_on = False
        if cap is not None:
            cap.release()
            cap = None  # Reset cap variable when camera is turned off
        return jsonify({"message": "Camera turned OFF"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
