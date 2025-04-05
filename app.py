from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# Load model and scaler
with open('sign_rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        decoded = base64.b64decode(image_data)
        img = Image.open(BytesIO(decoded)).convert('RGB')

        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Detect landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if landmarks:
                    landmarks_data = np.array(landmarks).reshape(1, -1)
                    landmarks_data = scaler.transform(landmarks_data)
                    prediction = rf_model.predict(landmarks_data)
                    return jsonify({'prediction': prediction[0]})

        return jsonify({'prediction': None})

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'prediction': None})

if __name__ == '__main__':
    app.run(debug=True)
