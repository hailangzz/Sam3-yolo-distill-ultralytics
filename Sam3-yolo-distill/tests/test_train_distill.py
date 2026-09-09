# -*- coding: utf-8 -*-

"""
Test full SAM3 -> YOLOv8-seg distillation pipeline

SAM3 Teacher
|
|
YOLO Student
|
|
Forward Hook
|
|
Adapter
|
|
FeatureLoss
|
|
Backward

"""

import torch

from ultralytics import YOLO

from teacher.sam3_teacher import SAM3Teacher

from modules.adapters import FeatureAdapter

from losses.feature_loss import FeatureLoss

from hooks.yolo_hook import YOLOFeatureHook



# =====================================================
# path
# =====================================================

YOLO_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/yolov8n-seg.pt"
)


SAM3_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/sam3.pt"
)


BPE_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/"
    "bpe_simple_vocab_16e6.txt.gz"
)



device = "cuda"



# =====================================================
# YOLO
# =====================================================

print("================")
print("load YOLO")


student = YOLO(
    YOLO_PATH
).model


student.cuda()

student.train()


print(
    "YOLO loaded"
)



# =====================================================
# hook
# =====================================================

print("================")
print("register hook")


hook = YOLOFeatureHook(

    student,

    layers=[
        15,
        18,
        21
    ]

)


print(
    "hook ready"
)



# =====================================================
# SAM3
# =====================================================

print("================")
print("load SAM3")


teacher = SAM3Teacher(

    SAM3_PATH,

    BPE_PATH,

    device=device,

    img_size=1008,

    fp16=True

)


print(
    "SAM3 ready"
)



# =====================================================
# Adapter
# =====================================================

print("================")
print("create adapter")


adapters = torch.nn.ModuleList(
    [

        FeatureAdapter(
            64,
            256
        ),


        FeatureAdapter(
            128,
            256
        ),


        FeatureAdapter(
            256,
            256
        )

    ]
)


adapters.cuda()


print(
    "adapter ready"
)



# =====================================================
# loss
# =====================================================

criterion = FeatureLoss()



# =====================================================
# optimizer
# =====================================================

optimizer = torch.optim.AdamW(

    list(student.parameters())

    +

    list(adapters.parameters()),

    lr=1e-4

)



# =====================================================
# fake image
# =====================================================

images = torch.randn(

    1,

    3,

    640,

    640,

    device=device

)



# =====================================================
# Teacher forward
# =====================================================

print("================")
print("teacher forward")


with torch.no_grad():

    teacher_features = teacher(
        images
    )


print(
    "SAM3 features"
)


for i,x in enumerate(
    teacher_features
):

    print(
        i,
        x.shape,
        x.dtype
    )



# =====================================================
# select SAM3 features for distillation
# =====================================================

teacher_distill = [

    teacher_features[1],

    teacher_features[2],

    teacher_features[3]

]



# =====================================================
# Student forward
# =====================================================

print("================")
print("student forward")


optimizer.zero_grad()

hook.clear()


output = student(
    images
)


print(
    "YOLO forward ok"
)



# =====================================================
# YOLO features
# =====================================================

print("================")
print("get YOLO features")


student_features = (

    hook.get_features()

)



for i,x in enumerate(
    student_features
):

    print(

        i,

        x.shape,

        x.dtype

    )



# =====================================================
# Adapter
# =====================================================

print("================")
print("adapter")


adapted = []


for i,x in enumerate(
    student_features
):


    y = adapters[i](

        x,

        target_size=
        teacher_distill[i].shape[-2:]

    )


    adapted.append(y)


    print(

        i,

        y.shape,

        y.dtype

    )



# =====================================================
# feature loss
# =====================================================

print("================")
print("feature loss")



#
# 注意顺序:
#
# 第一个参数:
#   student feature
#
# 第二个参数:
#   teacher feature
#

loss = criterion(

    adapted,

    teacher_distill

)



print(
    "loss:",
    loss
)


print(
    "loss requires grad:",
    loss.requires_grad
)



# =====================================================
# backward
# =====================================================

print("================")
print("backward")



loss.backward()


optimizer.step()



print(
    "backward success"
)



print("================")

print(
    "ALL TEST PASSED"
)

"""

cd /data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill
PYTHONPATH=/data/Sam3-yolo-distill-ultralytics:/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill python tests/test_train_distill.py

"""