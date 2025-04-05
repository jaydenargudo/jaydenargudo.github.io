from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load model and scaler
with open('sign_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        frame = np.array(img)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({'prediction': 'None'})

        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Predict
        X = scaler.transform([landmarks])
        prediction = model.predict(X)[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        print('Prediction error:', e)
        return jsonify({'prediction': 'Error'})

if __name__ == '__main__':
    app.run(debug=True)
