# -*- coding: utf-8 -*-

"""
SAM3 -> YOLOv8-seg Distillation Training

Pipeline:

Dataset
|
DataLoader
|
DistillTrainer
|
SAM3 Teacher
|
YOLO Student
|
Loss
|
Backward
|
Validation
|
mAP Evaluation
|
Checkpoint

"""


import os
import torch


from torch.utils.data import DataLoader


from train_distill import create_trainer


from datasets.yolo_dataset import YOLODataset


from validation.evaluator import DistillEvaluator


from validation.map_evaluator import MAPEvaluator


from utils.export_eval import (
    build_eval_model_from_student
)


from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint
)



# =====================================================
# config
# =====================================================


DEVICE = "cuda"



TRAIN_IMAGE_DIR = (
"/home/chenkejing/database/test/data/images/train"
)


TRAIN_LABEL_DIR = (
"/home/chenkejing/database/test/data/labels/train"
)



VAL_IMAGE_DIR = (
"/home/chenkejing/database/test/data/images/val"
)


VAL_LABEL_DIR = (
"/home/chenkejing/database/test/data/labels/val"
)



CHECKPOINT_DIR = (
"/data/Sam3-yolo-distill-ultralytics/"
"Sam3-yolo-distill/weights/checkpoints"
)



DATA_YAML = (
"/data/Sam3-yolo-distill-ultralytics/"
"Sam3-yolo-distill/datasets/carpet.yaml"
)



MODEL_YAML = (
"/data/Sam3-yolo-distill-ultralytics/ultralytics/cfg/models/v8/yolov8-seg.yaml"
)



os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)



# =====================================================
# resume
# =====================================================


RESUME = None



# =====================================================
# main
# =====================================================


def main():


    print("================")
    print("create trainer")


    trainer = create_trainer()


    print(
        "trainer ready"
    )



    # ==========================
    # resume
    # ==========================


    start_epoch = 0


    if RESUME is not None:


        start_epoch,_ = load_checkpoint(

            RESUME,

            trainer.student,

            trainer.adapters,

            trainer.optimizer,

            trainer.scaler

        )


        print(
            "resume epoch:",
            start_epoch
        )



    # ==========================
    # evaluator
    # ==========================


    print("================")
    print("create evaluator")


    evaluator = DistillEvaluator(

        trainer.teacher,

        trainer.student,

        trainer.adapters,

        trainer.feature_loss,

        trainer.yolo_hook,

        DEVICE

    )


    print(
        "evaluator ready"
    )



    # ==========================
    # dataset
    # ==========================


    train_dataset = YOLODataset(

        TRAIN_IMAGE_DIR,

        TRAIN_LABEL_DIR,

        640

    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=8,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

        collate_fn=train_dataset.collate_fn

    )



    val_dataset = YOLODataset(

        VAL_IMAGE_DIR,

        VAL_LABEL_DIR,

        640

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=8,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

        collate_fn=val_dataset.collate_fn

    )



    print(
        "train:",
        len(train_dataset)
    )


    print(
        "val:",
        len(val_dataset)
    )

    epochs = 100

    best_val_loss = float("inf")

    best_map = 0.0

    print("================")
    print("start training")

    # =================================================
    # epoch loop
    # =================================================

    for epoch in range(
            start_epoch,
            epochs
    ):

        print(
            "\nEpoch:",
            epoch
        )

        # ===============================
        # train mode
        # ===============================

        trainer.student.train()

        for adapter in trainer.adapters:
            adapter.train()

        epoch_loss = 0

        # ===============================
        # training
        # ===============================

        for step, batch in enumerate(train_loader):

            # ---------------------------
            # cuda
            # ---------------------------

            for k, v in batch.items():

                if torch.is_tensor(v):
                    batch[k] = v.cuda(

                        non_blocking=True

                    )

            # ---------------------------
            # train step
            # ---------------------------

            result = trainer.train_step(

                batch

            )

            epoch_loss += result["loss"]

            if step % 10 == 0:
                print(

                    "epoch:",
                    epoch,

                    "step:",
                    step,

                    result

                )

        avg_train_loss = (

                epoch_loss /

                len(train_loader)

        )

        print(

            "train loss:",

            avg_train_loss

        )

        # =================================================
        # validation loss
        # =================================================

        print("================")
        print("validation loss")

        val_result = evaluator.evaluate(

            val_loader

        )

        print(
            val_result
        )

        current_val_loss = (

            val_result["val_loss"]

        )

        # =================================================
        # mAP evaluation
        # =================================================

        print("================")
        print("mAP evaluation")

        # 创建临时YOLO模型用于val

        eval_model = build_eval_model_from_student(

            trainer.student,

            MODEL_YAML

        )

        map_evaluator = MAPEvaluator(

            eval_model,

            DATA_YAML,

            DEVICE

        )

        map_result = map_evaluator.evaluate()

        print(

            map_result

        )

        current_map = (

            map_result.get(

                "mask_mAP50_95",

                0

            )

        )

        # =================================================
        # save last checkpoint
        # =================================================

        last_path = os.path.join(

            CHECKPOINT_DIR,

            "last.pt"

        )

        save_checkpoint(

            last_path,

            epoch,

            trainer.student,

            trainer.adapters,

            trainer.optimizer,

            trainer.scaler,

            {

                "train_loss":

                    avg_train_loss,

                "val":

                    val_result,

                "map":

                    map_result

            }

        )

        print(

            "saved last checkpoint"

        )

        # =================================================
        # best validation loss
        # =================================================

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

            best_loss_path = os.path.join(

                CHECKPOINT_DIR,

                "best_loss.pt"

            )

            save_checkpoint(

                best_loss_path,

                epoch,

                trainer.student,

                trainer.adapters,

                trainer.optimizer,

                trainer.scaler,

                {

                    "train_loss":

                        avg_train_loss,

                    "val":

                        val_result,

                    "map":

                        map_result

                }

            )

            print(

                "new best val loss:",

                best_val_loss

            )

        # =================================================
        # best mAP
        # =================================================

        if current_map > best_map:
            best_map = current_map

            best_map_path = os.path.join(

                CHECKPOINT_DIR,

                "best_map.pt"

            )

            save_checkpoint(

                best_map_path,

                epoch,

                trainer.student,

                trainer.adapters,

                trainer.optimizer,

                trainer.scaler,

                {

                    "train_loss":

                        avg_train_loss,

                    "val":

                        val_result,

                    "map":

                        map_result

                }

            )

            print(

                "new best mAP:",

                best_map

            )

        # =================================================
        # periodic checkpoint
        # =================================================

        if epoch % 10 == 0:
            epoch_path = os.path.join(

                CHECKPOINT_DIR,

                f"epoch_{epoch}.pt"

            )

            save_checkpoint(

                epoch_path,

                epoch,

                trainer.student,

                trainer.adapters,

                trainer.optimizer,

                trainer.scaler,

                {

                    "train_loss":

                        avg_train_loss,

                    "val":

                        val_result,

                    "map":

                        map_result

                }

            )

            print(

                "saved:",

                epoch_path

            )

    print("================")
    print(
        "training finished"
    )


if __name__ == "__main__":
    main()

