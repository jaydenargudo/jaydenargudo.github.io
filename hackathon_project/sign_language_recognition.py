import cv2
import mediapipe as mp
import numpy as np
import pickle

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

# Buffer to hold landmarks for dynamic gesture recognition
landmark_sequence = []
max_frames = 10  # Number of frames to capture for dynamic gestures

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []

            # Extract landmarks (x, y, z)
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                z = lm.z
                landmarks.extend([x, y, z])

            # Add landmarks to the sequence
            landmark_sequence.append(landmarks)

            # If sequence exceeds the maximum length, remove the oldest frame
            if len(landmark_sequence) > max_frames:
                landmark_sequence.pop(0)

            # If we have enough frames, pass the sequence to the model
            if len(landmark_sequence) == max_frames:
                landmarks_data = np.array(landmark_sequence).reshape(1, -1)  # Flatten the landmarks
                landmarks_data = scaler.transform(landmarks_data)  # Normalize

                # Use the model to predict the gesture
                prediction = rf_model.predict(landmarks_data)

                # Display the predicted gesture
                cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw landmarks on the hand
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
