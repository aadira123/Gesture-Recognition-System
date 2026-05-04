import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load your trained YOLO model
model_path = 'best.pt'
try:
    model = YOLO(model_path)
    print(f"YOLO Model loaded successfully from {model_path}!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Try importing MediaPipe via explicit sub-modules to prevent import crashes
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ModuleNotFoundError:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

# 2. Initialize MediaPipe hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define clean drawing styles for the black-and-white skeleton
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# 3. Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("\nStarting Skeleton-based YOLO prediction feed. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Standardize orientation (No mirroring)
    h, w, c = frame.shape

    # 2. Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 3. Create a pure black canvas to replicate your training dataset style
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    label_to_display = "No hand detected"
    box_to_draw = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the white hand skeleton lines on the black canvas
            mp_drawing.draw_landmarks(
                black_canvas,
                hand_landmarks,
                HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), # White landmarks
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)  # White connections
            )

            # 4. Extract bounding box from the landmarks for better cropping
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Pad the bounding box
            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            # 5. Crop the black-and-white skeleton region
            skeleton_crop = black_canvas[y_min:y_max, x_min:x_max]

            if skeleton_crop.size > 0:
                # Resize cropped skeleton to match the model training size
                input_crop = cv2.resize(skeleton_crop, (640, 640))

                # 6. Make prediction using the YOLO model
                yolo_results = model(input_crop, conf=0.25, verbose=False)

                for r in yolo_results:
                    if len(r.boxes) > 0:
                        # Extract class name and confidence
                        cls_id = int(r.boxes.cls[0].cpu().numpy())
                        score = float(r.boxes.conf[0].cpu().numpy())
                        class_name = model.names[cls_id]

                        label_to_display = f"{class_name} ({score * 100:.1f}%)"
                        box_to_draw = (x_min, y_min, x_max, y_max)
                        break

    # Draw result on original camera frame for viewing
    if box_to_draw:
        x1, y1, x2, y2 = box_to_draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.putText(frame, f"Prediction: {label_to_display}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display real-time video frames
    cv2.imshow("Webcam View (Detections)", frame)
    cv2.imshow("What the Model Sees (White Skeleton on Black)", black_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()