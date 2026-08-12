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
Checkpoint

"""


import os
import torch

from torch.utils.data import DataLoader


from train_distill import create_trainer

from datasets.yolo_dataset import YOLODataset


from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint
)



# =====================================================
# config
# =====================================================


IMAGE_DIR = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/tests/data/images/train"
)


LABEL_DIR = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/tests/data/labels/train"
)



CHECKPOINT_DIR = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/weights/checkpoints"
)



os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)



# 是否恢复训练

RESUME = None


# 例如：
#
# RESUME = (
# "/data/ultralytics/"
# "Sam3-yolo-distill/weights/checkpoints/last.pt"
# )




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
    # resume
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



    # =================================================
    # dataset
    # =================================================


    print("================")
    print("create dataset")


    dataset = YOLODataset(

        image_dir=IMAGE_DIR,

        label_dir=LABEL_DIR,

        img_size=640

    )



    loader = DataLoader(

        dataset,

        batch_size=8,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

        collate_fn=dataset.collate_fn

    )



    print(
        "dataset:",
        len(dataset)
    )


    print(
        "batches:",
        len(loader)
    )



    # =================================================
    # training
    # =================================================


    epochs = 100



    print("================")
    print("start training")



    for epoch in range(
        start_epoch,
        epochs
    ):


        print(
            "\nEpoch:",
            epoch
        )



        epoch_loss = 0



        for step,batch in enumerate(loader):



            # ==========================
            # move cuda
            # ==========================


            for k,v in batch.items():


                if torch.is_tensor(v):


                    batch[k] = v.cuda(

                        non_blocking=True

                    )



            # ==========================
            # train step
            # ==========================


            result = trainer.train_step(
                batch
            )



            epoch_loss += (
                result["loss"]
            )



            if step % 10 == 0:


                print(

                    epoch,

                    step,

                    result

                )



        # ==========================
        # epoch loss
        # ==========================


        avg_loss = (

            epoch_loss /

            len(loader)

        )


        print(

            "epoch loss:",

            avg_loss

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

            avg_loss

        )



        # =================================================
        # save periodic checkpoint
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

                avg_loss

            )



    print("================")
    print(
        "training finished"
    )





if __name__=="__main__":

    main()