import math

from ultralytics import YOLO


model = YOLO("weights/yolov8n.pt", "v8")  # load a pretrained model (recommended for training)
results = model.predict("inference/images/dog05.jpeg", save=True)
object1 = None
object2 = None
for result in results:
    for box in result.boxes:
        class_name = model.names[int(box.cls)]
        if class_name == "dog":
            object1 = box.xyxy[0]
        if class_name == "couch":
            object2 = box.xyxy[0]
        print(model.names[int(box.cls)])
        print(box.xyxy[0])

if object1 is not None and object2 is not None:
    x1, y1, x2, y2 = object1
    x3, y3, x4, y4 = object2
    distance = ((x1 + x2) / 2 - (x3 + x4) / 2) ** 2 + ((y1 + y2) / 2 - (y3 + y4) / 2) ** 2
    distance = math.sqrt(distance)
    print("The distance between two objects:", distance)
    if distance < 100:
        print("Two objects are close to each other.")
    else:
        print("Two objects are far away from each other.")

    if (object1[0] < object2[2]) and (object1[2] > object2[0]) and (object1[1] < object2[3]) and (
            object1[3] > object2[1]):
        print("Two objects overlap.")
    else:
        print("Two objects do not overlap.")
