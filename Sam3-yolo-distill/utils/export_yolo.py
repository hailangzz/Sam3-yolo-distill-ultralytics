# -*- coding: utf-8 -*-

"""
Export distilled student
to Ultralytics YOLO format
"""


import torch

from ultralytics import YOLO



def export_student(

    checkpoint_path,

    model_yaml,

    output_path

):


    print("================")
    print("load checkpoint")



    checkpoint = torch.load(

        checkpoint_path,

        map_location="cpu"

    )



    print(
        "epoch:",
        checkpoint["epoch"]
    )



    # -----------------------
    # create YOLO model
    # -----------------------

    model = YOLO(
        model_yaml
    ).model



    # -----------------------
    # load student weights
    # -----------------------

    missing, unexpected = (

        model.load_state_dict(

            checkpoint["student"],

            strict=False

        )

    )



    print(
        "missing:",
        len(missing)
    )


    print(
        "unexpected:",
        len(unexpected)
    )



    # -----------------------
    # save
    # -----------------------

    torch.save(

        {

            "model":
            model,


            "epoch":
            checkpoint["epoch"],

        },

        output_path

    )



    print(
        "export:",
        output_path
    )