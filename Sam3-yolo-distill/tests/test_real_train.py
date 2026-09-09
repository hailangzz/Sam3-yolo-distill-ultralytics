# -*- coding: utf-8 -*-

"""
Real training pipeline

Dataset
|
DataLoader
|
DistillTrainer
|
SAM3 Teacher + YOLO Student
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


from datasets.yolo_dataset import YOLODataset


from train_distill import create_trainer



# =====================================================
# config
# =====================================================


# IMAGE_DIR = (
#     "/home/chenkejing/database/test/data/images/train"
# )
#
#
# LABEL_DIR = (
#     "/home/chenkejing/database/test/data/labels/train"
# )

IMAGE_DIR = (
    "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_carpet_database/images/train"
)


LABEL_DIR = (
    "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_carpet_database/labels/train"
)



DEVICE = "cuda"



EPOCHS = 100


BATCH_SIZE = 8



CHECKPOINT_DIR = (
    "./checkpoints"
)



os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)



# =====================================================
# checkpoint save
# =====================================================


def save_checkpoint(
        trainer,
        epoch,
        loss,
        filename
):


    save_path = os.path.join(
        CHECKPOINT_DIR,
        filename
    )


    checkpoint = {

        "epoch": epoch,

        "loss": loss,


        # student model
        "model": trainer.model.state_dict()
        if hasattr(trainer,"model")
        else None,

    }


    # optimizer

    if hasattr(
        trainer,
        "optimizer"
    ):

        checkpoint["optimizer"] = (
            trainer.optimizer.state_dict()
        )



    torch.save(
        checkpoint,
        save_path
    )


    print(
        "checkpoint saved:",
        save_path
    )





# =====================================================
# create trainer
# =====================================================


print("================")
print("create trainer")



trainer = create_trainer()



print(
    "trainer ready"
)





# =====================================================
# dataset
# =====================================================


print("================")
print("create dataset")



dataset = YOLODataset(

    image_dir=IMAGE_DIR,

    label_dir=LABEL_DIR,

    img_size=640

)



print(
    "dataset size:",
    len(dataset)
)



assert len(dataset)>0





# =====================================================
# dataloader
# =====================================================


loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    collate_fn=dataset.collate_fn

)



print(
    "dataloader ready"
)


print(
    "batch number:",
    len(loader)
)






# =====================================================
# training
# =====================================================


print("================")
print("START TRAINING")



best_loss = float("inf")



for epoch in range(
    1,
    EPOCHS + 1
):


    print("\n")
    print(
        "===================="
    )

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )



    total_loss = 0.0



    total_yolo = 0.0


    total_feature = 0.0



    batch_count = 0



    for batch_idx,batch in enumerate(loader):



        # -------------------------
        # cuda
        # -------------------------

        for k,v in batch.items():


            if torch.is_tensor(v):

                batch[k] = v.to(
                    DEVICE,
                    non_blocking=True
                )



        # -------------------------
        # train step
        # -------------------------


        result = trainer.train_step(
            batch
        )



        loss = result["loss"]



        total_loss += (
            float(loss)
        )


        total_yolo += (
            float(
                result["loss_yolo"]
            )
        )


        total_feature += (
            float(
                result["loss_feature"]
            )
        )



        batch_count += 1



        if batch_idx % 10 == 0:


            print(

                f"[Epoch {epoch}] "

                f"batch {batch_idx}/{len(loader)} "

                f"loss={loss:.4f} "

                f"yolo={result['loss_yolo']:.4f} "

                f"feature={result['loss_feature']:.4f}"

            )





    # =============================
    # epoch statistics
    # =============================


    avg_loss = (
        total_loss / batch_count
    )


    avg_yolo = (
        total_yolo / batch_count
    )


    avg_feature = (
        total_feature / batch_count
    )



    print(
        ""
    )

    print(
        f"Epoch {epoch} finished"
    )


    print(
        "loss:",
        avg_loss
    )


    print(
        "loss_yolo:",
        avg_yolo
    )


    print(
        "loss_feature:",
        avg_feature
    )




    # =============================
    # save epoch checkpoint
    # =============================


    save_checkpoint(

        trainer,

        epoch,

        avg_loss,

        f"epoch_{epoch}.pt"

    )



    # latest

    save_checkpoint(

        trainer,

        epoch,

        avg_loss,

        "latest.pt"

    )





    # best


    if avg_loss < best_loss:


        best_loss = avg_loss


        save_checkpoint(

            trainer,

            epoch,

            avg_loss,

            "best.pt"

        )



print("================")

print(
    "REAL TRAIN FINISHED"
)



print(
    "best loss:",
    best_loss
)