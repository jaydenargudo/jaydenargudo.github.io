import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# Initialize MediaPipe for hand landmarks detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
landmarks_data = []  # List to hold all the landmarks
labels = []  # List to hold corresponding labels for the gestures

# Add a path to save the training data
if not os.path.exists('data'):
    os.makedirs('data')

# Function to capture landmarks
def capture_landmarks():
    global landmarks_data, labels  # Declare these as global variables so we can use them outside the function
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return
    
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

                # Ask user to enter the gesture label
                cv2.putText(frame, "Press 's' to save data, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow("Capture Gesture", frame)

                key = cv2.waitKey(1)  # Wait for key press for 1 ms
                if key == ord('s'):  # Press 's' to save the current frame's data
                    label = input("Enter the label for this gesture: ")
                    landmarks_data.append(landmarks)
                    labels.append(label)
                    print(f"Gesture label {label} saved.")
                
                # Press 'q' to quit
                elif key == ord('q'):
                    print("Exiting capture...")
                    cap.release()  # Release the camera
                    cv2.destroyAllWindows()  # Close all OpenCV windows
                    return  # Break the loop and exit the function

# Function to train the model
def train_model():
    global landmarks_data, labels  # Make sure to use the global variables
    
    if len(landmarks_data) == 0 or len(labels) == 0:
        print("No data to train on.")
        return
    
    # Normalize the landmarks data
    scaler = StandardScaler()
    
    # Make sure the data is flattened to match the correct shape
    landmarks_data = np.array(landmarks_data)
    
    # Flatten the data for each gesture to have 630 features (10 frames * 63 landmarks per frame)
    landmarks_data = landmarks_data.reshape(landmarks_data.shape[0], -1)

    # Normalize the landmarks data using StandardScaler
    landmarks_data = scaler.fit_transform(landmarks_data)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(landmarks_data, labels)

    # Save the model and scaler
    with open('sign_rf_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Model and scaler saved!")

if __name__ == "__main__":
    capture_landmarks()  # Capture landmarks from the webcam
    train_model()  # Train the model with the captured data
