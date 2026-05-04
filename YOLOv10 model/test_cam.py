import cv2

# Initialize the video capture object. 
# 0 is the default index for built-in webcams.
cap = cv2.VideoCapture(0)

# Check if OpenCV successfully opened the camera
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    print("Troubleshooting tips:")
    print("1. Check if another application (Zoom, Teams, etc.) is using the camera.")
    print("2. Try changing the index from 0 to 1 or 2: cv2.VideoCapture(1)")
else:
    print("Webcam successfully connected!")

# Read a single frame as a quick sanity check
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Successfully opened the webcam, but failed to read a frame.")
else:
    print(f"Frame read successfully! Resolution: {frame.shape[1]}x{frame.shape[0]}")

print("\nStarting video stream window. Press 'q' to exit.")

# Display the stream in a loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show the feed
    cv2.imshow("Webcam Test", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()