import cv2
from ultralytics import YOLO

# 1. Load your custom emergency sign model
# Make sure best.pt is in the same directory, or provide the exact path
model = YOLO("best.pt")

# 2. Open the default webcam (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Set camera dimensions (optional, but optimizes speed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting webcam... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 3. Perform inference
    # stream=True uses a generator, making it highly memory efficient for video
    results = model(frame, stream=True)

    # 4. Draw bounding boxes and labels on the frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract Confidence and Class ID
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Filter low confidence detections (adjust threshold as needed)
            if conf > 0.5:
                # Class name from model
                class_name = model.names[cls_id]
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Overlay label and confidence score
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the real-time detection window
    cv2.imshow("Emergency Sign Detection - YOLO", frame)

    # Press 'q' to break the loop and quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()