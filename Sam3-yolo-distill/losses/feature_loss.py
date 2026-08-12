# -*- coding:utf-8 -*-

"""
Feature distillation loss

Teacher:
    SAM3

Student:
    YOLOv8

"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureLoss(nn.Module):


    def __init__(
        self,
        loss_type="mse"
    ):

        super().__init__()

        self.loss_type = loss_type



    def forward(
        self,
        student_features,
        teacher_features
    ):


        """
        Args:

        student_features:
            list[Tensor]


        teacher_features:
            list[Tensor]


        """

        assert len(student_features)==len(
            teacher_features
        )


        total_loss = 0.0



        for s,t in zip(
            student_features,
            teacher_features
        ):


            # teacher不参与梯度

            t = t.detach()
            t = t.float()


            # shape check

            if s.shape != t.shape:

                raise RuntimeError(
                    f"""
                    Feature shape mismatch:

                    student:
                    {s.shape}

                    teacher:
                    {t.shape}
                    """
                )



            if self.loss_type=="mse":


                loss = F.mse_loss(
                    s,
                    t
                )


            elif self.loss_type=="l1":


                loss = F.l1_loss(
                    s,
                    t
                )


            else:

                raise ValueError(
                    self.loss_type
                )



            total_loss += loss



        return total_loss / len(
            student_features
        )