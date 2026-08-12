# -*- coding:utf-8 -*-

import torch
import os



def save_checkpoint(
    path,
    epoch,
    student,
    adapters,
    optimizer,
    scaler,
    loss=None
):


    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )


    torch.save(
        {

            "epoch":
            epoch,


            "student":
            student.state_dict(),


            "adapters":
            adapters.state_dict(),


            "optimizer":
            optimizer.state_dict(),


            "scaler":
            scaler.state_dict(),


            "loss":
            loss

        },
        path
    )



def load_checkpoint(
    path,
    student,
    adapters,
    optimizer,
    scaler
):


    checkpoint=torch.load(
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


    scaler.load_state_dict(
        checkpoint["scaler"]
    )


    start_epoch = (
        checkpoint["epoch"]
        +
        1
    )


    return start_epoch