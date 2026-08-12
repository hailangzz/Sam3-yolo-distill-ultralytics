# -*- coding: utf-8 -*-

"""
Real training pipeline test

Dataset
|
DataLoader
|
DistillTrainer
|
SAM3 + YOLO
|
Loss
|
Backward
"""


import torch

from torch.utils.data import DataLoader


from datasets.yolo_dataset import YOLODataset


from train_distill import create_trainer



# =====================================================
# config
# =====================================================


IMAGE_DIR = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/tests/data/images/train"
)


LABEL_DIR = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/tests/data/labels/train"
)



DEVICE="cuda"



# =====================================================
# create trainer
# =====================================================


print("================")
print("create trainer")


trainer = create_trainer()



print(
    "trainer ready"
)



# =====================================================
# create dataset
# =====================================================


print("================")
print("create dataset")



dataset = YOLODataset(

    image_dir=IMAGE_DIR,

    label_dir=LABEL_DIR,

    img_size=640

)



print(
    "dataset size:",
    len(dataset)
)



assert len(dataset)>0



# =====================================================
# dataloader
# =====================================================


loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True,

    num_workers=0,

    collate_fn=dataset.collate_fn

)



print(
    "dataloader ready"
)


print(
    "batch number:",
    len(loader)
)



# =====================================================
# get batch
# =====================================================


print("================")
print("get batch")



batch = next(
    iter(loader)
)



for k,v in batch.items():


    if torch.is_tensor(v):

        print(
            k,
            v.shape,
            v.dtype,
            v.device
        )


    else:

        print(
            k,
            type(v)
        )



# =====================================================
# move cuda
# =====================================================


print("================")
print("move batch cuda")



for k,v in batch.items():


    if torch.is_tensor(v):

        batch[k]=v.to(
            DEVICE,
            non_blocking=True
        )



print(
    "batch cuda ready"
)



for k,v in batch.items():

    if torch.is_tensor(v):

        print(
            k,
            v.device
        )



# =====================================================
# train step
# =====================================================


print("================")
print("run train step")



result = trainer.train_step(
    batch
)



print(
    result
)



# =====================================================
# check loss
# =====================================================


print("================")
print("check loss")



assert "loss" in result

assert "loss_yolo" in result

assert "loss_feature" in result



assert result["loss"] > 0

assert result["loss_yolo"] > 0

assert result["loss_feature"] > 0



print("================")

print(
    "REAL TRAIN TEST PASSED"
)
