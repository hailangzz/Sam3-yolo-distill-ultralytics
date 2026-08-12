# -*- coding: utf-8 -*-

"""
Checkpoint utilities

Save:
    YOLO student
    Adapter
    Optimizer
    AMP scaler

Resume training
"""


import os
import torch



# =====================================
# save checkpoint
# =====================================

def save_checkpoint(
    path,
    epoch,
    student,
    adapters,
    optimizer,
    scaler=None,
    loss=None
):


    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )


    checkpoint = {


        "epoch":
        epoch,


        "student":
        student.state_dict(),


        "adapters":
        adapters.state_dict(),


        "optimizer":
        optimizer.state_dict(),


        "loss":
        loss

    }



    if scaler is not None:

        checkpoint["scaler"] = (
            scaler.state_dict()
        )



    torch.save(

        checkpoint,

        path

    )



    print(
        "checkpoint saved:",
        path
    )





# =====================================
# load checkpoint
# =====================================

def load_checkpoint(
    path,
    student,
    adapters,
    optimizer,
    scaler=None
):


    checkpoint = torch.load(

        path,

        map_location="cuda"

    )



    student.load_state_dict(

        checkpoint["student"]

    )



    adapters.load_state_dict(

        checkpoint["adapters"]

    )



    optimizer.load_state_dict(

        checkpoint["optimizer"]

    )



    if (
        scaler is not None
        and
        "scaler" in checkpoint
    ):

        scaler.load_state_dict(

            checkpoint["scaler"]

        )



    start_epoch = (

        checkpoint["epoch"]

        +

        1

    )


    last_loss = checkpoint.get(
        "loss",
        None
    )


    print(
        "resume checkpoint:",
        path
    )

    print(
        "resume epoch:",
        start_epoch
    )



    return (
        start_epoch,
        last_loss
    )