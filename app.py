from ultralytics import YOLO
import cv2
import pandas as pd
import os
from datetime import datetime
import time

model = YOLO("yolov8n.pt")

os.makedirs(
    "captures",
    exist_ok=True
)

log_file = "detection_history.csv"

if not os.path.exists(
    log_file
):

    pd.DataFrame(
        columns=[
            "Time",
            "Object"
        ]
    ).to_csv(
        log_file,
        index=False
    )

cap = cv2.VideoCapture(0)

confidence = 0.6

previous = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(
        frame,
        conf=confidence
    )

    output = results[0].plot()

    counts = {}

    for box in results[0].boxes:

        cls = int(
            box.cls
        )

        name = model.names[
            cls
        ]

        counts[name] = (
            counts.get(
                name,
                0
            )
            + 1
        )

    y = 40

    for k,v in counts.items():

        text = (
            f"{k}: {v}"
        )

        cv2.putText(
            output,
            text,
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        y += 40

        pd.DataFrame(
            [
                [
                    datetime.now(),
                    k
                ]
            ]
        ).to_csv(
            log_file,
            mode="a",
            header=False,
            index=False
        )

    current = time.time()

    fps = int(
        1/(current-previous)
    )

    previous = current

    cv2.putText(
        output,
        f"FPS: {fps}",
        (20,400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.imshow(
        "AI Object Detection",
        output
    )

    key = cv2.waitKey(1)

    if key == ord("s"):

        name = (
            "captures/"
            +
            datetime.now().strftime(
                "%H%M%S"
            )
            +
            ".jpg"
        )

        cv2.imwrite(
            name,
            output
        )

        print(
            "Saved:",
            name
        )

    elif key == 27:

        break

cap.release()

cv2.destroyAllWindows()