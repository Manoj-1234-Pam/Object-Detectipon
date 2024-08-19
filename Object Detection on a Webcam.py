import cv2
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the webcam (0 for the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the output frame
    cv2.imshow("YOLOv8 Detection - Webcam", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close display windows
cap.release()
cv2.destroyAllWindows()
