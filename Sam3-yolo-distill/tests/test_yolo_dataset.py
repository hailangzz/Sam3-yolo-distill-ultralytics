# -*- coding: utf-8 -*-

"""
Test YOLO Dataset

Verify:

dataset
    |
    |
load image
    |
    |
load label
    |
    |
return YOLO batch format

"""

import os
import torch

from datasets.yolo_dataset import YOLODataset



# =====================================================
# dataset path
# =====================================================

IMAGE_DIR = (
    "/home/chenkejing/database/test/data/images/train"
)

LABEL_DIR = (
    "/home/chenkejing/database/test/data/labels/train"
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
    "dataset length:",
    len(dataset)
)



assert len(dataset) > 0



# =====================================================
# get sample
# =====================================================

print("================")
print("get sample")



sample = dataset[0]



print(sample.keys())



# =====================================================
# check image
# =====================================================

img = sample["img"]



print(
    "image:",
    img.shape,
    img.dtype
)



assert isinstance(
    img,
    torch.Tensor
)



assert img.shape[0] == 3



# =====================================================
# check labels
# =====================================================

print("================")
print("check labels")



print(
    "cls:",
    sample["cls"]
)



print(
    "bboxes:",
    sample["bboxes"]
)



assert "cls" in sample

assert "bboxes" in sample



# =====================================================
# check batch format
# =====================================================

print("================")
print("batch format")



from torch.utils.data import DataLoader


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=dataset.collate_fn
)


batch = next(iter(loader))


for k,v in batch.items():

    if isinstance(v, torch.Tensor):

        print(
            k,
            v.shape,
            v.dtype
        )

    else:

        print(
            k,
            type(v)
        )



assert "img" in batch

assert "batch_idx" in batch



print("================")
print(
    "YOLO DATASET TEST PASSED"
)

"""
cd /data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill

PYTHONPATH=/data/Sam3-yolo-distill-ultralytics:/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill \
python tests/test_yolo_dataset.py
"""