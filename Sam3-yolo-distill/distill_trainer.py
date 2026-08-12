# -*- coding: utf-8 -*-

"""
SAM3 Feature Distillation Trainer

Teacher:
SAM3

Student:
YOLOv8-seg

AMP training enabled
"""


import torch

from torch.cuda.amp import (
    autocast,
    GradScaler
)



class DistillTrainer:


    def __init__(
        self,
        teacher,
        student,
        adapters,
        feature_loss,
        optimizer,
        yolo_hook,
        lambda_feature=1.0,
        device="cuda"
    ):


        self.device = device


        self.teacher = teacher

        self.student = student

        self.adapters = adapters


        self.feature_loss = feature_loss


        self.optimizer = optimizer


        self.yolo_hook = yolo_hook


        self.lambda_feature = lambda_feature



        # =====================================
        # AMP scaler
        # =====================================

        self.scaler = GradScaler()



        # =====================================
        # freeze SAM3
        # =====================================

        self.teacher.eval()


        for p in self.teacher.parameters():

            p.requires_grad = False



        # =====================================
        # train mode
        # =====================================

        self.student.train()


        for adapter in self.adapters:

            adapter.train()



    # =========================================
    # train step
    # =========================================


    def train_step(
        self,
        batch
    ):



        self.optimizer.zero_grad()



        images = batch["img"]



        # =====================================
        # 1. SAM3 teacher
        # =====================================


        with torch.no_grad():


            teacher_features = self.teacher(
                images
            )



        teacher_features = [

            x.detach()

            for x in teacher_features

        ]



        sam3_features = [

            teacher_features[1],

            teacher_features[2],

            teacher_features[3]

        ]



        # =====================================
        # 2. YOLO forward
        # =====================================


        self.yolo_hook.clear()



        with autocast():


            student_output = self.student(
                images
            )



        student_features = (
            self.yolo_hook.get_features()
        )



        if len(student_features) != 3:


            raise RuntimeError(
                f"YOLO feature number error: {len(student_features)}"
            )



        # =====================================
        # 3. Adapter
        # =====================================


        adapted_features = []



        with autocast():


            for i, feature in enumerate(
                student_features
            ):


                y = self.adapters[i](

                    feature,

                    target_size =
                    sam3_features[i].shape[-2:]

                )


                adapted_features.append(y)



        # =====================================
        # 4. Feature loss
        #
        # MSE建议float32
        # =====================================


        adapted_features_fp32 = [

            x.float()

            for x in adapted_features

        ]


        sam3_features_fp32 = [

            x.float()

            for x in sam3_features

        ]



        loss_feature = self.feature_loss(

            adapted_features_fp32,

            sam3_features_fp32

        )



        # =====================================
        # 5. YOLO segmentation loss
        # =====================================


        with autocast():


            yolo_result = self.student.loss(
                batch
            )



        if isinstance(
            yolo_result,
            tuple
        ):


            loss_yolo = yolo_result[0]


        else:


            loss_yolo = yolo_result



        # YOLO loss scalar

        if loss_yolo.ndim > 0:

            loss_yolo = loss_yolo.sum()



        # =====================================
        # 6. total loss
        # =====================================


        loss = (

            loss_yolo

            +

            self.lambda_feature
            *
            loss_feature

        )



        if loss.ndim > 0:

            loss = loss.sum()



        # =====================================
        # AMP backward
        # =====================================


        self.scaler.scale(
            loss
        ).backward()



        self.scaler.step(
            self.optimizer
        )


        self.scaler.update()



        return {


            "loss":

            loss.detach().item(),



            "loss_yolo":

            loss_yolo.detach().item(),



            "loss_feature":

            loss_feature.detach().item()

        }