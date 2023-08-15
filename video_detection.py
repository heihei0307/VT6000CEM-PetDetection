import random
from ultralytics import YOLO
import cv2
import math
from datetime import datetime

mainDetect = ["dog", "cat"]
dangerDetect = []
defaultMode = 0

inputMode = int(input("Choose the Mode (Video - 0, Webcam - 1): ") or defaultMode)

danger = ''
while danger.strip() == '':
    danger = input("Please input the danger area: ")

danger = danger.split(",")
dangerDetect = danger

model = YOLO("weights/yolov8m.pt", "v8")

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.splitlines()
my_file.close()

# Generate random colors for class list
class_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    class_colors.append((b, g, r))


cap = cv2.VideoCapture("inference/videos/dog004.mp4")
if inputMode == 1:
    cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

current_dateTime = datetime.now()
newDate = str(current_dateTime.year) + str(current_dateTime.month) + str(current_dateTime.day) + '_' + str(
    current_dateTime.hour) + str(current_dateTime.minute)

# to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('runs/videos/output' + newDate + '.mp4', fourcc, int(fps), (int(width), int(height)))

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    success, frame = cap.read()
    results = model(frame, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes
        mainObject = []
        dangerObject = []
        result = []

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print(" --- Confidence --- ", confidence)

            # class name
            cls = int(box.cls[0])
            print(" --- Class name --- ", class_list[cls])

            if class_list[cls] in mainDetect:
                mainObject.append({"name": class_list[cls], "obj": box.xyxy[0]})

            if class_list[cls] in dangerDetect:
                dangerObject.append({"name": class_list[cls], "obj": box.xyxy[0]})

            if class_list[cls] in mainDetect or class_list[cls] in dangerDetect:
                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[cls], 3)

                # object details
                org = [x1, y1+25]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = class_colors[cls]
                thickness = 2

                cv2.putText(frame, class_list[cls], org, font, fontScale, color, thickness)

        for dangerItem in dangerObject:
            for mainItem in mainObject:
                obj = {"main": "", "danger": "", "distance": "", "overlap": False}
                font = cv2.FONT_HERSHEY_SIMPLEX
                x1, y1, x2, y2 = mainItem["obj"]
                x3, y3, x4, y4 = dangerItem["obj"]
                mainName = mainItem["name"]
                dangerName = dangerItem["name"]
                obj["main"] = mainName
                obj["danger"] = dangerName

                distance = ((x1 + x2) / 2 - (x3 + x4) / 2) ** 2 + ((y1 + y2) / 2 - (y3 + y4) / 2) ** 2
                distance = math.sqrt(distance)
                obj["distance"] = distance
                # print(f"The distance between {mainName} and {dangerName}:", distance)
                # if distance < 100:
                #     print(f"{mainName} and {dangerName} are close to each other.")
                # else:
                #     print(f"{mainName} and {dangerName} are far away from each other.")

                if (x1 < x4) and (x2 > x3) and (y1 < y4) and (y2 > y3):
                    # print(f"{mainName} and {dangerName} overlap.")
                    obj["overlap"] = True
                    cv2.putText(
                        frame, f"{mainName} and {dangerName} overlap.",
                        (50, 50), font, 1, (102, 102, 255), 2,
                    )
                # else:
                #     print(f"{mainName} and {dangerName} do not overlap.")
                result.append(obj)

        print(result)

    cv2.imshow('Video', frame)
    output_frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    writer.write(output_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()