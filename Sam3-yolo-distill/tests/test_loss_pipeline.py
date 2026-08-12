# -*- coding:utf-8 -*-

import torch

from train_distill import trainer


device="cuda"


print("================")
print("create fake batch")


print("================")
print("create fake batch")


batch = {

    # image
    "img":
    torch.randn(
        1,
        3,
        640,
        640,
        device="cuda"
    ),


    # object belongs to image 0
    "batch_idx":
    torch.tensor(
        [0],
        device="cuda"
    ),


    # class id
    "cls":
    torch.tensor(
        [
            [0]
        ],
        dtype=torch.float32,
        device="cuda"
    ),


    # bbox xywh
    # normalized

    "bboxes":
    torch.tensor(
        [
            [
                0.5,
                0.5,
                0.2,
                0.2
            ]
        ],
        dtype=torch.float32,
        device="cuda"
    ),


    # segmentation mask

    "masks":
    torch.zeros(
        1,
        640,
        640,
        device="cuda"
    )

}


# create fake object mask

batch["masks"][

    0,

    200:400,

    200:400

]=1


result = trainer.train_step(
    batch
)


print(result)

"""
cd /data/ultralytics/Sam3-yolo-distill

PYTHONPATH=/data/ultralytics:/data/ultralytics/Sam3-yolo-distill \
python tests/test_loss_pipeline.py
"""