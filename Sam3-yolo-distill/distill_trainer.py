# -*- coding: utf-8 -*-

"""
SAM3 Feature Distillation Trainer

Teacher:
SAM3 text conditioned

Student:
YOLOv8-seg

"""


import torch

from torch.cuda.amp import autocast, GradScaler


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
        device="cuda",
    ):

        self.device = device

        self.teacher = teacher

        self.student = student

        self.adapters = adapters

        self.feature_loss = feature_loss

        self.optimizer = optimizer

        self.yolo_hook = yolo_hook

        self.lambda_feature = lambda_feature

        self.scaler = GradScaler()

        # freeze SAM3

        self.teacher.eval()

        for p in self.teacher.parameters():

            p.requires_grad = False

        self.student.train()

        for adapter in self.adapters:

            adapter.train()

    # ==================================================
    # build prompt batch
    # ==================================================

    def build_prompt_batch(self, batch):

        """
        Convert instance prompts to image prompts

        input:

        prompts:
        [
          "carpet",
          "wire",
          "liquid"
        ]

        batch_idx:
        tensor([
            0,
            0,
            1
        ])


        output:

        [
          [
            "carpet",
            "wire"
          ],

          [
            "liquid"
          ]
        ]

        """

        prompts = batch["prompts"]

        batch_idx = batch["batch_idx"]

        batch_size = batch["img"].shape[0]

        image_prompts = []

        for img_id in range(batch_size):

            current_prompts = []

            ids = torch.where(
                batch_idx == img_id
            )[0]

            for idx in ids:
                idx = idx.item()

                current_prompts.append(
                    prompts[idx]
                )

            # 去重
            current_prompts = list(
                set(current_prompts)
            )

            # 空目标
            if len(current_prompts) == 0:
                current_prompts.append(
                    "background"
                )

            image_prompts.append(
                current_prompts
            )

        return image_prompts

    # ==================================================
    # SAM3 forward
    # ==================================================

    def forward_teacher(
            self,
            images,
            prompt_batch
    ):

        batch_outputs = []

        B = images.shape[0]

        for i in range(B):

            img = images[i:i + 1]

            prompt_features = []

            for prompt in prompt_batch[i]:
                feature = self.teacher(

                    img,

                    [prompt]

                )

                prompt_features.append(
                    feature
                )

            fused = []

            for level in range(
                    len(prompt_features[0])
            ):

                level_features = []

                for f in prompt_features:
                    level_features.append(
                        f[level]
                    )

                merged = torch.cat(

                    level_features,

                    dim=0

                )

                #
                # text conditioned feature fusion
                #

                level_feature = torch.max(

                    merged,

                    dim=0,

                    keepdim=True

                )[0]

                fused.append(
                    level_feature
                )

            batch_outputs.append(
                fused
            )

        outputs = []

        for level in range(
                len(batch_outputs[0])
        ):
            outputs.append(

                torch.cat(

                    [
                        x[level]
                        for x in batch_outputs
                    ],

                    dim=0

                )

            )

        return outputs

    # ==================================================
    # train step
    # ==================================================

    def train_step(self, batch):

        self.optimizer.zero_grad()

        images = batch["img"]

        # =============================
        # prompt
        # =============================

        prompt_batch = self.build_prompt_batch(batch)

        # =============================
        # SAM3 teacher
        # =============================

        with torch.no_grad():

            teacher_features = self.forward_teacher(images, prompt_batch)

        teacher_features = [x.detach() for x in teacher_features]

        sam3_features = [teacher_features[1], teacher_features[2], teacher_features[3]]

        # =============================
        # YOLO
        # =============================

        self.yolo_hook.clear()

        with autocast():

            self.student(images)

        student_features = self.yolo_hook.get_features()

        if len(student_features) != 3:

            raise RuntimeError("YOLO feature number error")

        # =============================
        # adapter
        # =============================

        adapted = []

        with autocast():

            for i, f in enumerate(student_features):

                adapted.append(
                    self.adapters[i](f, target_size=sam3_features[i].shape[-2:])
                )

        adapted = [x.float() for x in adapted]

        sam3_features = [x.float() for x in sam3_features]

        loss_feature = self.feature_loss(adapted, sam3_features)

        # =============================
        # YOLO loss
        # =============================

        with autocast():

            yolo_result = self.student.loss(batch)

        if isinstance(yolo_result, tuple):

            loss_yolo = yolo_result[0]

        else:

            loss_yolo = yolo_result

        if loss_yolo.ndim > 0:

            loss_yolo = loss_yolo.sum()

        loss = loss_yolo + self.lambda_feature * loss_feature

        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)

        self.scaler.update()

        return {
            "loss": loss.item(),
            "loss_yolo": loss_yolo.item(),
            "loss_feature": loss_feature.item(),
        }
