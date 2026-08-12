# -*- coding: utf-8 -*-

"""
SAM3 -> YOLOv8-seg Feature Distillation Training

Teacher:
    SAM3 Vision Encoder

Student:
    YOLOv8-seg

"""


import torch
from ultralytics import YOLO


from teacher.sam3_teacher import SAM3Teacher

from modules.adapters import FeatureAdapter

from losses.feature_loss import FeatureLoss

from distill_trainer import DistillTrainer

from hooks.yolo_hook import YOLOHook




# =====================================================
# path
# =====================================================


YOLO_PATH = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/yolov8n-seg.pt"
)


SAM3_PATH = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/models/sam3.pt"
)


BPE_PATH = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/models/"
    "bpe_simple_vocab_16e6.txt.gz"
)



DEVICE="cuda"



# =====================================================
# Load YOLO Student
# =====================================================


print("================")
print("loading YOLO")


student = YOLO(
    YOLO_PATH
).model



student.cuda()


student.train()



print(student)



# =====================================================
# YOLO Hook
# =====================================================


print("================")
print("register YOLO hook")


hook = YOLOHook(
    student
)


hook.register()



print("hook ready")





# =====================================================
# Load SAM3 Teacher
# =====================================================


print("================")
print("loading SAM3")


teacher = SAM3Teacher(

    SAM3_PATH,

    BPE_PATH,

    device=DEVICE,

    img_size=1008,

    fp16=True

)



print("SAM3 ready")





# =====================================================
# Feature Adapter
# =====================================================


print("================")
print("build adapters")


adapters = [

    FeatureAdapter(
        128,
        256
    ),


    FeatureAdapter(
        256,
        256
    ),


    FeatureAdapter(
        512,
        256
    )

]



adapters = [

    x.cuda().half()

    for x in adapters

]



print("adapter ready")





# =====================================================
# Loss
# =====================================================


feature_loss = FeatureLoss()



# =====================================================
# Optimizer
# =====================================================


params=[]


# YOLO parameters

params += list(
    student.parameters()
)



# adapter parameters


for adapter in adapters:

    params += list(
        adapter.parameters()
    )



optimizer = torch.optim.AdamW(

    params,

    lr=1e-4,

    weight_decay=5e-4

)




# =====================================================
# Trainer
# =====================================================


trainer = DistillTrainer(

    teacher,

    student,

    adapters,

    feature_loss,

    optimizer,

    lambda_feature=1.0,

    device=DEVICE

)



print("================")
print("trainer ready")





# =====================================================
# dummy dataloader
#
# 后续替换成 YOLO Dataset
# =====================================================



for epoch in range(10):


    print(
        "epoch",
        epoch
    )



    images = torch.randn(

        1,
        3,
        1008,
        1008,

        device=DEVICE

    )



    result = trainer.train_step(

        images

    )



    print(result)





print("training finished")