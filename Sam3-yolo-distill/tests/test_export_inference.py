# -*- coding: utf-8 -*-

from ultralytics import YOLO


MODEL = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/"
    "weights/"
    "distilled_yolov8n_seg.pt"
)


IMAGE = (
    "/home/chenkejing/database/test/data/images/train/image_batch1_14.jpg"
)


print("================")
print("load distilled model")


model = YOLO(
    MODEL
)


print(
    "model loaded"
)



print("================")
print("predict")


results = model.predict(

    IMAGE,

    imgsz=640,

    conf=0.25,

    save=True

)



print(
    "detections:",
    len(results)
)


print("================")
print(
    "EXPORT INFERENCE TEST PASSED"
)