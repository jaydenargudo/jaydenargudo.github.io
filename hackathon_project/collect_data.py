import cv2
import mediapipe as mp
import csv

# Setup MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

cap = cv2.VideoCapture(0)

label = input("Enter the label for this sign (e.g., A): ")
data = []

print("[INFO] Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame. Exiting...")
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Get image dimensions (height, width)
    h, w, c = frame.shape  # c is the number of channels (3 for RGB)

    # Convert image to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        # Draw the hand landmarks on the frame
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Debugging: Print how many hands are detected
        # print(f"[DEBUG] Detected {len(result.multi_hand_landmarks)} hand(s)")

        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []

            # Extract and scale landmarks
            for lm in hand_landmarks.landmark:
                x = lm.x * w  # Scale x to image width
                y = lm.y * h  # Scale y to image height
                z = lm.z * w  # Scale z to image width (or height, or leave unscaled)

                landmarks.extend([x, y, z])

            landmarks.append(label)  # Add the label (sign)

            # Wait for key press after the landmarks are extracted
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Save data when 's' is pressed
                print(f"[DEBUG] Saved landmarks for {label}: {landmarks}")
                print(f"[DEBUG] Data length after saving: {len(data)}")
                data.append(landmarks)  # Save the data for the sign
            elif key == ord('q'):  # Quit when 'q' is pressed
                break

    # Show the label on screen
    cv2.putText(frame, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame with hand landmarks
    cv2.imshow("Data Collection", frame)

    # Exit condition if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save data to CSV
with open('sign_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("[DONE] Data saved to sign_data.csv")
