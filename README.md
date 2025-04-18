
#  Mental Health Chatbot

A compassionate, AI-powered mental health support system built to assist users in expressing their emotions and guiding them towards self-care, relaxation, and support.

# Features

 Chatbot with Empathetic Responses
- Built using **Rasa** for natural language understanding and dialogue management.
- Handles mood-specific responses (happy, sad, anxious, depressed, etc.)
- Responds sensitively to critical phrases like **"I want to end it"**, offering real support and helpline info.

 Mood Tracking & Visualization
- Detects mood using **TextBlob sentiment analysis** and **keyword-based emotion mapping**.
- Stores mood logs in **SQLite** only when the mood changes.
- Displays mood history through daily, weekly, and monthly charts.

 Silent SOS Mode
- Monitors user messages for critical SOS phrases like *“help me”*, *“suicidal”*, *“end myself”*.
- Logs SOS events and responds with emergency support resources like the **KIRAN Mental Health Helpline** (1800-599-0019).

 Safe Space Drawing Board
- A creative outlet where users can draw with a pen, marker, highlighter, and erase freely.
- Fully interactive drawing area with vertical tool selector on the side.

 Message to Future Self
- Users can write a personal note to their future selves.
- Displays the message back later as a reminder of growth or hope.

 Self-Care Hub
- Curated **motivational and yoga video links**.
- A quick **Mental Health Quiz** with tailored suggestions.
- Includes a collection of **poems and stories** for emotional nourishment.


# Tech Stack

| Layer        | Tools Used                                     |
|--------------|------------------------------------------------|
| Frontend     | HTML, Tailwind CSS, JavaScript                 |
| Backend      | Flask (Python), SQLite                         |
| Chatbot Core | Rasa (NLU + Dialogue)                          |
| Audio Input  | Whisper (Speech-to-Text)                       |
| Emotion Detection | DeepFace + OpenCV                         |
| Mood Detection | TextBlob (Sentiment Analysis)                |

---

# Project Structure

```
Mental_health_chatbot/
├── backend/
│   ├── app.py                # Flask app logic
│   ├── templates/            # HTML templates
│   └── static/               # CSS, JS, images
├── rasa/                     # Rasa project (NLU, domain, stories)
├── users.db                  # SQLite DB for users & mood logs
└── README.md


# Installation

1. Clone the repository:

git clone https://github.com/yourusername/mental-health-chatbot.git
cd mental-health-chatbot

2. Create and activate a virtual environment:

python -m venv rasa_env
rasa_env\Scripts\activate  # Windows

3. Install dependencies:

pip install -r requirements.txt


4. Train the Rasa model:

rasa train


5. Start the Flask app:

python app.py


6. Start the Rasa server:

rasa run --enable-api


# Emergency Support

This chatbot is not a replacement for professional help.

**If you’re in crisis, please call the KIRAN Mental Health Helpline: `1800-599-0019`**


# Created By

Rafiya Iqbal
Sanjana Shaji
Shahanaz Fathima
Raniya Iqbal


