import cv2
from ultralytics import YOLO

# 1. Load your trained YOLO model
# Replace 'best.pt' with the correct path to your YOLO model file
model_path = 'best.pt' 

try:
    model = YOLO(model_path)
    print(f"YOLO Model loaded successfully from {model_path}!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# 2. Open the webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("\nStarting YOLO webcam prediction feed. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Flip the frame for a mirrored view
    frame = cv2.flip(frame, 1)

    # 3. Run YOLO inference on the current frame
    # verbose=False suppresses the console printout for every frame
    results = model(frame, verbose=False)

    # 4. Iterate over detection results
    for result in results:
        # Get bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        names = result.names  # Get the class names mapping from the model

        for box, score, class_id in zip(boxes, scores, class_ids):
            # Only display detections above a confidence threshold (e.g., 40%)
            if score >= 0.40:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box)
                
                # Get label name and confidence score
                label_name = names[int(class_id)]
                label_text = f"{label_name} ({score * 100:.1f}%)"

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Add the label text above the bounding box
                cv2.putText(
                    frame, 
                    label_text, 
                    (x1, max(y1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )

    # 5. Display the frame with predictions
    cv2.imshow("Emergency Sign Language Detection - YOLO", frame)

    # Press 'q' to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()