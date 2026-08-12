# -*- coding: utf-8 -*-

"""
Convert trained student model
to Ultralytics YOLO object
"""

import torch

from ultralytics import YOLO



def build_eval_model(
    checkpoint_path,
    model_yaml
):

    print("================")
    print("build eval model")


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )


    model = YOLO(
        model_yaml
    )


    model.model.load_state_dict(
        checkpoint["student"],
        strict=False
    )


    print(
        "eval model ready"
    )


    return model