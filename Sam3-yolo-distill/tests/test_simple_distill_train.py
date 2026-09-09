# -*- coding:utf-8 -*-

"""
Simple SAM3 -> YOLO Feature Distillation

Purpose:

verify whether SAM3 feature
can supervise YOLO feature

No prompt
No adapter
No validation

"""

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from ultralytics import YOLO



from datasets.yolo_dataset import YOLODataset

from teacher.sam3_teacher import SAM3Teacher

from hooks.yolo_hook import YOLOFeatureHook



DEVICE="cuda"


IMAGE_DIR="/data/database/jrdb_yolo_random_val/images/val"

LABEL_DIR="/data/database/jrdb_yolo_random_val/labels/val"



# ============================
# config
# ============================

EPOCHS=20

BATCH=4



# ============================
# build YOLO
# ============================


print("load student")


student = YOLO(
    "yolov8n-seg.pt"
).model



student.to(
    DEVICE
)



student.train()



# ============================
# hook
# ============================


print("create hook")


hook = YOLOFeatureHook(
    student
)



# ============================
# SAM3
# ============================


print("load SAM3")


teacher = SAM3Teacher(

    model_path=
    "models/sam3.pt",

    bpe_path=
    "models/bpe_simple_vocab_16e6.txt.gz",

    device=DEVICE,

    fp16=True

)



# ============================
# dataset
# ============================


dataset = YOLODataset(

    image_dir=IMAGE_DIR,

    label_dir=LABEL_DIR,

    img_size=640

)


loader = DataLoader(

    dataset,

    batch_size=BATCH,

    shuffle=True,

    num_workers=2,

    collate_fn=
    dataset.collate_fn

)



# ============================
# optimizer
# ============================


optimizer=torch.optim.AdamW(

    student.parameters(),

    lr=1e-4

)



# ============================
# train
# ============================


for epoch in range(EPOCHS):


    total=0


    for step,batch in enumerate(loader):


        images=batch["img"].to(
            DEVICE
        )


        # -------------------
        # SAM3 feature
        # -------------------

        with torch.no_grad():

            sam3_feature = teacher(
                images,
                [
                    "object"
                ]
            )



        # -------------------
        # YOLO feature
        # -------------------


        hook.clear()


        student(
            images
        )


        yolo_feature = (
            hook.get_features()
        )



        # -------------------
        # simple feature loss
        # -------------------


        loss=0


        for sf,tf in zip(
            yolo_feature,
            sam3_feature
        ):


            tf=F.interpolate(

                tf,

                size=sf.shape[-2:]

            )


            loss += F.mse_loss(

                sf,

                tf.detach()

            )



        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total+=loss.item()



        if step%10==0:


            print(

                epoch,

                step,

                loss.item()

            )


    print(
        "epoch loss:",
        total/len(loader)
    )



print("finish")