from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
with open('sign_rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize MediaPipe Hands for hand landmarks detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Function to generate video feed for web display
def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    landmarks.extend([x, y, z])

                # Make prediction if landmarks are available
                if landmarks:
                    landmarks_data = np.array(landmarks).reshape(1, -1)
                    landmarks_data = scaler.transform(landmarks_data)
                    prediction = rf_model.predict(landmarks_data)
                    cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Index route to render the webcam page
@app.route('/')
def index():
    return render_template('index.html')

# Route to return video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to trigger data collection (e.g., save landmarks and labels)
@app.route('/collect_data', methods=['POST'])
def collect_data():
    label = request.form['label']
    # Assuming you have the capture_landmarks() function or similar for saving data
    # Implement saving data here
    return 'Data collected'

# Run the app with Gunicorn on Render, and fallback to Flask locally
if __name__ == '__main__':
    # On Render, it will pick up the `PORT` environment variable automatically
    # For local testing, default to port 8000 if PORT environment variable is not set
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)
