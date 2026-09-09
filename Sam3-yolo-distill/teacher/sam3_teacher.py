# -*- coding: utf-8 -*-

"""
SAM3 Teacher Wrapper

SAM3 multimodal teacher

Input:

images:
    Tensor[B,3,H,W]

prompts:
    list[str]


Output:

SAM3 text conditioned feature pyramid


Used for:

SAM3 -> YOLOv8-seg feature distillation

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



        # =====================================
        # load SAM3
        # =====================================

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


        sam3_model = predictor.model


        sam3_model.to(device)


        sam3_model.eval()



        self.model = sam3_model



        # =====================================
        # SAM3 components
        # =====================================


        self.vision_encoder = (

            sam3_model
            .backbone
            .vision_backbone

        )


        self.language_encoder = (

            sam3_model
            .backbone
            .language_backbone

        )



        self.vision_encoder.set_imgsz(

            [
                img_size,
                img_size
            ]

        )



        # =====================================
        # freeze
        # =====================================


        for p in self.parameters():

            p.requires_grad = False



        if fp16:

            self.half()



    # =====================================
    # forward
    # =====================================


    @torch.no_grad()
    def forward(
        self,
        images,
        prompts
    ):


        """
        images:

            Tensor[B,3,H,W]


        prompts:

            list[str]


        example:

            [
                "carpet"
            ]

        """



        # =================================
        # image preprocess
        # =================================


        images = images.to(
            self.device
        )


        if self.fp16:

            images = images.half()



        if (

            images.shape[-1]
            !=
            self.img_size

            or

            images.shape[-2]
            !=
            self.img_size

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



        # =================================
        # vision feature
        # =================================


        vision_output = self.vision_encoder(

            images

        )


        #
        # SAM3:
        #
        # (
        #   feature pyramid,
        #   position,
        #   ...
        # )
        #

        vision_features = vision_output[0]



        # =================================
        # text feature
        # =================================


        text_outputs = self.language_encoder(

            prompts

        )


        """
        SAM3 language output:


        (
            token_count,

            token_features,

            global_features
        )


        token_features:

            [32,1,256]


        global_features:

            [32,1,1024]

        """



        token_features = text_outputs[1]



        #
        # [32,1,256]
        #
        # ->
        #
        # [1,256]
        #

        text_features = token_features.mean(

            dim=0

        )



        #
        # debug
        #

        # print(
        #     "text feature:",
        #     text_features.shape
        # )



        # =================================
        # image-text fusion
        # =================================


        fused_features = []



        for feat in vision_features:



            #
            # text:
            #
            # [1,256]
            #
            # ->
            #
            # [1,256,1,1]
            #

            text = (

                text_features

                .unsqueeze(-1)

                .unsqueeze(-1)

            )



            #
            # broadcast
            #

            feat = feat + text



            fused_features.append(

                feat

            )



        return fused_features