# -*- coding: utf-8 -*-

"""
SAM3 -> YOLOv8-seg Feature Distillation

Only build trainer

No training loop here

"""


import torch


from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace


from teacher.sam3_teacher import SAM3Teacher

from modules.adapters import FeatureAdapter

from losses.feature_loss import FeatureLoss

from distill_trainer import DistillTrainer

from hooks.yolo_hook import YOLOFeatureHook



# =====================================================
# config
# =====================================================


DEVICE = "cuda"



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



# =====================================================
# create trainer
# =====================================================


def create_trainer():


    # =====================================
    # YOLO student
    # =====================================

    print("================")
    print("load YOLO")


    student = YOLO(
        YOLO_PATH
    ).model



    # fix args

    if isinstance(
        student.args,
        dict
    ):

        student.args = IterableSimpleNamespace(
            **student.args
        )



    #
    # segmentation loss params
    #

    student.args.overlap_mask = True

    student.args.mask_ratio = 4

    student.args.box = 7.5

    student.args.cls = 0.5

    student.args.dfl = 1.5



    student.cuda()

    student.train()



    print(
        "YOLO loaded"
    )



    # =====================================
    # YOLO hook
    # =====================================


    print("================")
    print("register YOLO hook")


    hook = YOLOFeatureHook(

        student,

        layers=[
            15,
            18,
            21
        ]

    )


    hook.register()



    print(
        "hook ready"
    )



    # =====================================
    # SAM3 teacher
    # =====================================


    print("================")
    print("load SAM3")


    teacher = SAM3Teacher(

        SAM3_PATH,

        BPE_PATH,

        device=DEVICE,

        img_size=1008,

        fp16=True

    )



    print(
        "SAM3 ready"
    )



    # =====================================
    # adapter
    # =====================================


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



    # =====================================
    # feature loss
    # =====================================


    feature_loss = FeatureLoss()



    # =====================================
    # optimizer
    # =====================================


    params = []


    params += list(
        student.parameters()
    )


    params += list(
        adapters.parameters()
    )



    optimizer = torch.optim.AdamW(

        params,

        lr=1e-4,

        weight_decay=5e-4

    )



    # =====================================
    # trainer
    # =====================================


    trainer = DistillTrainer(

        teacher,

        student,

        adapters,

        feature_loss,

        optimizer,

        hook,

        lambda_feature=1.0,

        device=DEVICE

    )


    print("================")
    print("trainer ready")



    return trainer