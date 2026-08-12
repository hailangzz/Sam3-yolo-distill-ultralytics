# -*- coding: utf-8 -*-

"""
YOLOv8-seg mAP Evaluator

Evaluate:
    box mAP
    mask mAP

Used for:
    baseline YOLO
    SAM3 distillation YOLO
"""


import torch

from ultralytics import YOLO



class MAPEvaluator:


    def __init__(
        self,
        model,
        data_yaml,
        device="cuda"
    ):

        self.model = model

        self.data_yaml = data_yaml

        self.device = device



    # =====================================
    # evaluate
    # =====================================

    @torch.no_grad()
    def evaluate(self):


        print("================")
        print("start mAP evaluation")



        results = self.model.val(

            data=self.data_yaml,

            device=self.device,

            verbose=False

        )



        metrics = {}



        # ===============================
        # box metrics
        # ===============================

        if hasattr(
            results,
            "box"
        ):

            metrics.update(

                {

                    "box_mAP50":

                    float(
                        results.box.map50
                    ),


                    "box_mAP50_95":

                    float(
                        results.box.map
                    )

                }

            )



        # ===============================
        # mask metrics
        # ===============================

        if hasattr(
            results,
            "seg"
        ):

            metrics.update(

                {

                    "mask_mAP50":

                    float(
                        results.seg.map50
                    ),


                    "mask_mAP50_95":

                    float(
                        results.seg.map
                    )

                }

            )



        print("================")
        print("mAP result")


        for k,v in metrics.items():

            print(
                k,
                ":",
                v
            )


        return metrics