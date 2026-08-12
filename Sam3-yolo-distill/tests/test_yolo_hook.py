import torch

from ultralytics import YOLO

from hooks.yolo_hook import YOLOFeatureHook



# ======================
# load YOLOv8-seg
# ======================

model = YOLO(
    "yolov8n-seg.pt"
)

model.model.cuda()

model.model.eval()



# ======================
# register hook
# ======================

hook = YOLOFeatureHook(
    model.model,
    layers=[
        15,
        18,
        21
    ]
)



# ======================
# fake image
# ======================

image = torch.randn(
    1,
    3,
    640,
    640
).cuda()



with torch.no_grad():

    output = model.model(
        image
    )



# ======================
# check feature
# ======================


features = hook.get_features()


print("================")

print(
    len(features)
)



for i,f in enumerate(features):

    print(
        i,
        f.shape
    )

hook.remove()

# PYTHONPATH=/data/ultralytics:/data/ultralytics/Sam3-yolo-distill  python tests/test_yolo_hook.py