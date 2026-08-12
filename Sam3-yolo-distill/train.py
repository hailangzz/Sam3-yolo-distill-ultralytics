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
Checkpoint

"""


import os
import torch


from torch.utils.data import DataLoader


from train_distill import create_trainer


from datasets.yolo_dataset import YOLODataset


from validation.evaluator import DistillEvaluator


from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint
)



# =====================================================
# config
# =====================================================


DEVICE = "cuda"



TRAIN_IMAGE_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/images/train"
)


TRAIN_LABEL_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/labels/train"
)



VAL_IMAGE_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/images/val"
)


VAL_LABEL_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/labels/val"
)



CHECKPOINT_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/weights/checkpoints"
)



os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)



# =====================================================
# resume
# =====================================================


RESUME = None


# example:
#
# RESUME =
# "/data/ultralytics/Sam3-yolo-distill/weights/checkpoints/last.pt"



# =====================================================
# main
# =====================================================


def main():


    # =================================================
    # create trainer
    # =================================================


    print("================")
    print("create trainer")


    trainer = create_trainer()


    print(
        "trainer ready"
    )



    # =================================================
    # resume checkpoint
    # =================================================


    start_epoch = 0



    if RESUME is not None:


        start_epoch, _ = load_checkpoint(

            RESUME,

            trainer.student,

            trainer.adapters,

            trainer.optimizer,

            trainer.scaler

        )


        print(
            "resume from epoch:",
            start_epoch
        )



    # =================================================
    # evaluator
    # =================================================


    print("================")
    print("create evaluator")


    evaluator = DistillEvaluator(

        trainer.teacher,

        trainer.student,

        trainer.adapters,

        trainer.feature_loss,

        trainer.yolo_hook,

        device=DEVICE

    )


    print(
        "evaluator ready"
    )



    # =================================================
    # train dataset
    # =================================================


    print("================")
    print("create train dataset")



    train_dataset = YOLODataset(

        image_dir=TRAIN_IMAGE_DIR,

        label_dir=TRAIN_LABEL_DIR,

        img_size=640

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=8,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

        collate_fn=train_dataset.collate_fn

    )



    print(
        "train dataset:",
        len(train_dataset)
    )



    # =================================================
    # validation dataset
    # =================================================


    print("================")
    print("create val dataset")



    val_dataset = YOLODataset(

        image_dir=VAL_IMAGE_DIR,

        label_dir=VAL_LABEL_DIR,

        img_size=640

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
        "val dataset:",
        len(val_dataset)
    )



    # =================================================
    # training config
    # =================================================


    epochs = 100


    best_val_loss = float("inf")



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



        trainer.student.train()

        for adapter in trainer.adapters:

            adapter.train()



        epoch_loss = 0



        # ===============================
        # train
        # ===============================


        for step,batch in enumerate(train_loader):


            for k,v in batch.items():


                if torch.is_tensor(v):

                    batch[k] = v.cuda(

                        non_blocking=True

                    )



            result = trainer.train_step(

                batch

            )



            epoch_loss += result["loss"]



            if step % 10 == 0:


                print(

                    epoch,

                    step,

                    result

                )



        avg_loss = (

            epoch_loss /

            len(train_loader)

        )


        print(

            "train loss:",

            avg_loss

        )



        # ===============================
        # validation
        # ===============================


        print("================")
        print("validation")



        val_result = evaluator.evaluate(

            val_loader

        )


        print(

            val_result

        )



        current_val_loss = (

            val_result["val_loss"]

        )



        # ===============================
        # save last
        # ===============================


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
                avg_loss,

                "val":
                val_result

            }

        )



        print(
            "saved last checkpoint"
        )



        # ===============================
        # save best
        # ===============================


        if current_val_loss < best_val_loss:


            best_val_loss = current_val_loss



            best_path = os.path.join(

                CHECKPOINT_DIR,

                "best.pt"

            )



            save_checkpoint(

                best_path,

                epoch,

                trainer.student,

                trainer.adapters,

                trainer.optimizer,

                trainer.scaler,

                {

                    "train_loss":
                    avg_loss,

                    "val":
                    val_result

                }

            )



            print(

                "new best model:",

                best_val_loss

            )



        # ===============================
        # periodic checkpoint
        # ===============================


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
                    avg_loss,

                    "val":
                    val_result

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