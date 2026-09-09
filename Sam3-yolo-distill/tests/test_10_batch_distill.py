# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader


from datasets.yolo_dataset import YOLODataset

from teacher.sam3_teacher import SAM3Teacher

from modules.adapters import build_adapters

from losses.feature_loss import FeatureLoss

from hooks.yolo_hook import YOLOHook

from distill_trainer import DistillTrainer


from ultralytics import YOLO



# ==========================
# config
# ==========================


device="cuda"


image_dir="/your/train/images"

label_dir="/your/train/labels"



# ==========================
# dataset
# ==========================


dataset = YOLODataset(

    image_dir,

    label_dir,

    img_size=640

)



loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True,

    num_workers=4,

    collate_fn=dataset.collate_fn

)



# ==========================
# YOLO student
# ==========================


student = YOLO(
    "yolov8n-seg.pt"
).model



student.to(device)



# ==========================
# hook
# ==========================


yolo_hook = YOLOHook(
    student
)



# ==========================
# SAM3
# ==========================


teacher = SAM3Teacher(

    model_path=
    "models/sam3.pt",

    bpe_path=
    "models/bpe_simple_vocab_16e6.txt.gz",

    device=device

)



# ==========================
# adapter
# ==========================


adapters = build_adapters()



for x in adapters:

    x.to(device)



# ==========================
# loss
# ==========================


feature_loss = FeatureLoss()



# ==========================
# optimizer
# ==========================


optimizer=torch.optim.AdamW(

    list(student.parameters())

    +

    list(
        adapters.parameters()
    ),

    lr=1e-4

)



# ==========================
# trainer
# ==========================


trainer=DistillTrainer(

    teacher,

    student,

    adapters,

    feature_loss,

    optimizer,

    yolo_hook,

    lambda_feature=1.0,

    device=device

)



# ==========================
# train 10 batch
# ==========================


student.train()



for step,batch in enumerate(loader):


    print("======================")

    print(
        "step:",
        step
    )


    # move tensor


    for k,v in batch.items():

        if torch.is_tensor(v):

            batch[k]=v.to(device)



    result = trainer.train_step(
        batch
    )


    print(result)



    if step>=9:

        break



print(
    "10 batch test finished"
)