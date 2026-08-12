# -*- coding: utf-8 -*-

"""
SAM3 Teacher Wrapper

用于 YOLOv8-Seg feature distillation

Input:
    images:
        Tensor[B,3,H,W]

Output:
    SAM3 feature pyramid:
        [
            Tensor[B,256,288,288],
            Tensor[B,256,144,144],
            Tensor[B,256,72,72],
            Tensor[B,256,36,36]
        ]

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


from ultralytics.models.sam import SAM3SemanticPredictor



class SAM3Teacher(nn.Module):

    def __init__(
        self,
        model_path,
        bpe_path,
        device="cuda",
        img_size=1008,
        fp16=True,
    ):

        super().__init__()


        self.device = device

        self.img_size = img_size

        self.fp16 = fp16


        # ==========================
        # SAM3 predictor
        # ==========================

        overrides = {

            "conf":0.25,

            "task":"segment",

            "mode":"predict",

            "model":model_path,

        }


        predictor = SAM3SemanticPredictor(
            overrides=overrides
        )


        predictor.bpe_path = bpe_path


        predictor.setup_model()


        model = predictor.model


        model.to(device)

        model.eval()



        # 保存 vision backbone

        self.backbone = (
            model
            .backbone
            .vision_backbone
        )


        # 设置 SAM3 输入尺寸

        self.backbone.set_imgsz(
            [
                img_size,
                img_size
            ]
        )



        # ==========================
        # freeze
        # ==========================

        for p in self.parameters():

            p.requires_grad=False



        if fp16:

            self.half()



    @torch.no_grad()
    def forward(
        self,
        images
    ):


        """
        images:

        Tensor:
            [B,3,H,W]

        range:
            0~1

        """



        # ==========================
        # resize
        # ==========================

        if (
            images.shape[-1] != self.img_size
            or
            images.shape[-2] != self.img_size
        ):


            images = F.interpolate(
                images,
                size=(
                    self.img_size,
                    self.img_size
                ),
                mode="bilinear",
                align_corners=False
            )



        images = images.to(
            self.device
        )


        if self.fp16:

            images = images.half()



        # ==========================
        # SAM3 forward
        # ==========================

        outputs = self.backbone(
            images
        )


        """
        outputs:

        (
            sam3_features,
            sam3_pos,
            sam2_features,
            sam2_pos
        )

        """


        sam3_features = outputs[0]


        return sam3_features