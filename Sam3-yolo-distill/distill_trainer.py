# -*- coding: utf-8 -*-

"""
SAM3 Feature Distillation Trainer

Teacher:
SAM3

Student:
YOLOv8-seg
"""

import torch



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



        # =========================
        # freeze SAM3
        # =========================

        self.teacher.eval()


        for p in self.teacher.parameters():

            p.requires_grad = False



        # =========================
        # train mode
        # =========================

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
        # 1. SAM3 teacher feature
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
        # 3. adapter
        # =====================================


        adapted_features = []


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
        # =====================================


        loss_feature = self.feature_loss(

            adapted_features,

            sam3_features

        )



        # =====================================
        # 5. YOLO native segmentation loss
        # =====================================


        yolo_result = self.student.loss(
            batch
        )



        #
        # Ultralytics compatibility
        #

        if isinstance(
            yolo_result,
            tuple
        ):


            loss_yolo = yolo_result[0]

        else:


            loss_yolo = yolo_result



        #
        # make scalar
        #

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



        #
        # ensure scalar
        #

        if loss.ndim > 0:

            loss = loss.sum()



        # =====================================
        # backward
        # =====================================


        loss.backward()


        self.optimizer.step()



        return {


            "loss":

            loss.detach().item(),



            "loss_yolo":

            loss_yolo.detach().item(),



            "loss_feature":

            loss_feature.detach().item()

        }