import cv2
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Load an image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Draw bounding boxes and labels on the image
annotated_frame = results[0].plot()

# Display the output
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image
output_path = 'output_image.jpg'
cv2.imwrite(output_path, annotated_frame)
