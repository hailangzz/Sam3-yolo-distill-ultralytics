# -*- coding:utf-8 -*-

"""
Feature adapters for SAM3 -> YOLO distillation
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):

    """
    Convert teacher feature channels
    to student feature channels
    """


    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.adapter = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)

        )

    def forward(self, x, target_size=None):
        x = self.adapter(x)

        if target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

        return x



class MultiScaleFeatureAdapter(nn.Module):

    """
    SAM3 has 4 scales

    Convert:

    [
      256,
      256,
      256,
      256
    ]

    to YOLO channels
    """


    def __init__(
        self,
        teacher_channels,
        student_channels
    ):

        super().__init__()


        assert len(teacher_channels)==len(student_channels)


        self.adapters = nn.ModuleList()


        for t,s in zip(
            teacher_channels,
            student_channels
        ):

            self.adapters.append(

                FeatureAdapter(
                    t,
                    s
                )

            )



    def forward(
        self,
        features
    ):


        outputs=[]


        for adapter,feat in zip(
            self.adapters,
            features
        ):

            outputs.append(
                adapter(feat)
            )


        return outputs