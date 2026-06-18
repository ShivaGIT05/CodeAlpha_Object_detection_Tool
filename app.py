from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True
    )

    output = results[0].plot()

    cv2.imshow(
        "Object Detection & Tracking",
        output
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()