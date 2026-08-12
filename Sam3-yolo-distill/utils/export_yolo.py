# -*- coding: utf-8 -*-

"""
Export distilled YOLOv8-seg student

Remove:
SAM3
Adapter
Distillation modules

Keep:
YOLO student only

用于导出部署的模型
"""


import torch

from ultralytics import YOLO

from ultralytics.utils import IterableSimpleNamespace



def export_student(

        checkpoint_path,

        model_path,

        output_path

):


    print("================")
    print("load checkpoint")


    checkpoint = torch.load(

        checkpoint_path,

        map_location="cpu"

    )


    print(

        "checkpoint epoch:",

        checkpoint["epoch"]

    )



    # ==================================
    # load original YOLO model
    # ==================================


    print("================")
    print("load YOLO model")


    yolo = YOLO(

        model_path

    )


    model = yolo.model



    if isinstance(

        model.args,

        dict

    ):


        model.args = IterableSimpleNamespace(

            **model.args

        )



    print(
        "YOLO loaded"
    )



    # ==================================
    # load distilled weights
    # ==================================


    print("================")
    print("load student weights")



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



    # ==================================
    # build ultralytics checkpoint
    # ==================================


    print("================")
    print("save YOLO checkpoint")



    model.cpu()

    model.eval()



    torch.save(

        {


            "model":

            model,


            "ema":

            None,


            "epoch":

            checkpoint["epoch"],



        },

        output_path

    )



    print(

        "export success:",

        output_path

    )



if __name__ == "__main__":


    export_student(

        checkpoint_path=

        "/data/ultralytics/"
        "Sam3-yolo-distill/"
        "weights/checkpoints/best.pt",



        model_path=

        "/data/ultralytics/"
        "Sam3-yolo-distill/"
        "yolov8n-seg.pt",



        output_path=

        "/data/ultralytics/"
        "Sam3-yolo-distill/"
        "weights/"
        "distilled_yolov8n_seg.pt"

    )