# -*- coding: utf-8 -*-

"""
Test SAM3 text condition effect

Verify:

same image
different prompts

whether SAM3 feature changes

"""

import torch

import os
import sys


# =====================================
# project path
# =====================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT)



from teacher.sam3_teacher import SAM3Teacher



# =====================================
# config
# =====================================

DEVICE = "cuda"


SAM3_MODEL = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/sam3.pt"
)


BPE_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/"
    "bpe_simple_vocab_16e6.txt.gz"
)



IMAGE_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/tests/test.jpg"
)



# =====================================
# load image
# =====================================

import cv2
import torch.nn.functional as F


def load_image(path):


    img = cv2.imread(path)


    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    img = cv2.resize(
        img,
        (640,640)
    )


    img = (

        torch.from_numpy(img)

        .permute(
            2,
            0,
            1
        )

        .float()

        /
        255.0

    )


    return img.unsqueeze(0)



# =====================================
# main
# =====================================


if __name__ == "__main__":


    print("================")
    print("load SAM3 teacher")


    teacher = SAM3Teacher(

        model_path=SAM3_MODEL,

        bpe_path=BPE_PATH,

        device=DEVICE,

        fp16=True

    )


    teacher.eval()



    image = load_image(
        IMAGE_PATH
    )


    image = image.to(
        DEVICE
    )



    # =================================
    # prompt 1
    # =================================


    print("================")
    print("prompt: carpet")


    feature_carpet = teacher(

        image,

        [
            "carpet"
        ]

    )



    # =================================
    # prompt 2
    # =================================


    print("================")
    print("prompt: wire")


    feature_wire = teacher(

        image,

        [
            "wire"
        ]

    )



    # =================================
    # compare
    # =================================


    print("================")
    print("compare features")


    for i,(a,b) in enumerate(
        zip(
            feature_carpet,
            feature_wire
        )
    ):


        diff = torch.mean(

            torch.abs(

                a.float()

                -

                b.float()

            )

        )


        cosine = torch.nn.functional.cosine_similarity(

            a.flatten(),

            b.flatten(),

            dim=0

        )



        print(
            "level:",
            i
        )

        print(
            "shape:",
            a.shape
        )


        print(
            "mean abs diff:",
            diff.item()
        )


        print(
            "cosine:",
            cosine.item()
        )



    print("================")
    print("done")