# Import necessary libraries
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("yolov8m.pt")

# Run prediction on the image
results = model.predict("dog.jpg")
result = results[0]

# Print the number of detected objects
print(len(result.boxes))

# Process each detected box
for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    conf = round(box.conf[0].item(), 2)
    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")

# Display the image with detected objects
Image.fromarray(result.plot()[:, :, ::-1])
