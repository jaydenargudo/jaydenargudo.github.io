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
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        image_np = np.array(image)

        # MediaPipe processes BGR images (OpenCV format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = hands.process(image_bgr)

        if not results.multi_hand_landmarks:
            print("[INFO] No hand detected.")
            return jsonify({'prediction': 'No hand detected'})

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) != 63:
                print(f"[ERROR] Unexpected landmark length: {len(landmarks)}")
                return jsonify({'prediction': 'Invalid landmark data'})

            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks_scaled = scaler.transform(landmarks)
            prediction = rf_model.predict(landmarks_scaled)[0]
            print(f"[PREDICTION] {prediction}")
            return jsonify({'prediction': prediction})

        return jsonify({'prediction': 'Hand not recognized'})

    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({'error': 'Prediction error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
