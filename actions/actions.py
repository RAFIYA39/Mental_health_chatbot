from transformers import pipeline
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import random
import torch
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from rasa_sdk.events import SlotSet


# Load models once (optimization)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
chatbot_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=0 if torch.cuda.is_available() else -1)

class ActionGenerateEmpatheticResponse(Action):
    def name(self):
        return "action_generate_empathetic_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_message = tracker.latest_message.get("text").lower()
        detected_emotion = tracker.get_slot("emotion")  # Get detected emotion

        # Emotion-based response mapping
        emotion_responses = {
            "happy": "I'm glad you're feeling happy! 😊",
            "sad": "I'm sorry you're feeling down. Do you want to talk about it?",
            "angry": "It sounds like you're upset. I'm here to listen.",
            "stressed": "I'm sorry you're feeling overwhelmed. Would you like to try a calming exercise?",
            "neutral": None  # Let AI generate a response for neutral cases
        }
        
        response = emotion_responses.get(detected_emotion)

        # Perform sentiment analysis
        sentiment_result = sentiment_pipeline(user_message)[0]
        sentiment_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]

        # Handling high-confidence negative sentiment
        if sentiment_label == "NEGATIVE" and sentiment_score > 0.85:
            response = random.choice([
                "I'm really sorry you're feeling this way. You're not alone. Do you want to talk about it?",
                "That sounds tough. I'm here to listen. What's on your mind?",
                "It’s okay to feel this way. Would you like to do a calming exercise?"
            ])

        # Ensure AI response is empathetic
        if not response:
            ai_response = chatbot_pipeline(user_message, max_length=100, do_sample=True, temperature=0.7)[0]["generated_text"]

            # Prevent inappropriate AI responses
            if any(word in ai_response.lower() for word in ["great", "awesome", "carry on"]):
                response = "That sounds difficult. I'm here to support you."
            else:
                response = ai_response

        dispatcher.utter_message(response)
        return []

class ActionCheerUp(Action):
    def name(self):
        return "action_cheer_up"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        cheer_up_messages = [
            "Here’s a joke for you: Why don’t skeletons fight each other? They don’t have the guts! 😂",
            "How about listening to your favorite music? 🎶",
            "Take a deep breath, you're stronger than you think! 💪"
        ]
        dispatcher.utter_message(random.choice(cheer_up_messages))
        return []

class ActionAcknowledgeEmotion(Action):
    def name(self):
        return "action_acknowledge_emotion"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_message = tracker.latest_message.get("text")

        # Perform sentiment analysis
        sentiment_result = sentiment_pipeline(user_message)[0]
        sentiment_label = sentiment_result["label"]

        # Provide empathetic responses
        if sentiment_label == "NEGATIVE":
            response = "It sounds like you're going through a tough time. I'm here to listen. 💙"
        elif sentiment_label == "POSITIVE":
            response = "That's great to hear! Keep up the positivity! 😊"
        else:
            response = "I'm here to chat with you. Tell me more about how you're feeling."

        dispatcher.utter_message(response)
        return []

class ActionHandleCriticalEmotions(Action):
    def name(self):
        return "action_handle_critical_emotions"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_message = tracker.latest_message.get("text").lower()

        # List of high-risk phrases
        crisis_phrases = ["i want to die", "i feel suicidal", "i hate my life", "i can't do this anymore"]

        if any(phrase in user_message for phrase in crisis_phrases):
            dispatcher.utter_message("I'm really sorry you're feeling this way. You're not alone. Please consider talking to a trusted friend, family member, or a mental health professional. 💙")
            dispatcher.utter_message("Would you like me to suggest some resources that might help?")
            return []

        return []

class ActionDetectFacialEmotion(Action):
    def name(self):
        return "action_detect_facial_emotion"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)  # Open webcam

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                        face = frame[y:y + h, x:x + w]

                        # Perform facial emotion recognition
                        try:
                            emotion_analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                            emotion = emotion_analysis[0]['dominant_emotion']
                            print(f"Detected Emotion: {emotion}")
                            
                            # Pass emotion to Rasa
                            return [SlotSet("emotion", emotion)]
                        except:
                            pass

                cv2.imshow("Facial Emotion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        dispatcher.utter_message("Emotion detection stopped.")
        return []