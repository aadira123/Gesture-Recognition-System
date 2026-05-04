
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load your trained CNN model
try:
    model = tf.keras.models.load_model('emergency_sign_model.keras')
    print("CNN Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Class labels (alphabetical order matching training)
CLASSES = ['accident', 'call', 'doctor', 'help', 'hot', 'lose', 'pain', 'thief']

# 2. Configure MediaPipe Hand Landmarker using the new Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("\nStarting webcam prediction feed. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Flip the frame horizontally for a natural mirrored view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # MediaPipe Tasks require RGB images wrapped in their own Image object
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect landmarks using the Tasks API
    detection_result = detector.detect(mp_image)

    label = "No hand detected"
    confidence = 0.0

    # If hands are found
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Get the exact crop bounding box coordinates for hand tracking
            x_coordinates = [lm.x for lm in hand_landmarks]
            y_coordinates = [lm.y for lm in hand_landmarks]

            x_min = int(min(x_coordinates) * w)
            x_max = int(max(x_coordinates) * w)
            y_min = int(min(y_coordinates) * h)
            y_max = int(max(y_coordinates) * h)

            # Add padding to capture full context around hand gestures
            padding = 25
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Draw a simple bounding box around the hand on screen
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract the Region of Interest (ROI)
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess crop for CNN inference
            if hand_roi.size > 0:
                img_resized = cv2.resize(hand_roi, (224, 224))
                
                # Rescale raw image to match model inputs (normalization)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_array = np.array(img_rgb) / 255.0
                img_array = np.expand_dims(img_array, axis=0) # Dimensions match (1, 224, 224, 3)

                # Make the prediction
                predictions = model.predict(img_array, verbose=0)
                pred_idx = np.argmax(predictions[0])
                confidence = predictions[0][pred_idx]

                # Adjust confidence threshold as needed
                if confidence > 0.40:
                    label = f"{CLASSES[pred_idx]} ({confidence*100:.1f}%)"
                else:
                    label = "Uncertain"

    # Display prediction result text directly on the live feed
    cv2.putText(frame, f"Prediction: {label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Emergency Sign Language Detection - CNN", frame)

    # Press 'q' to close the stream window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()