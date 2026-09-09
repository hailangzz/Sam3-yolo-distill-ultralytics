# -*- coding: utf-8 -*-

"""
Test SAM3 text grounding inference

SAM3:
    image + text prompt

Output:
    masks
    boxes
    scores

"""

import os
import cv2
import torch
import numpy as np

from ultralytics.models.sam import SAM3SemanticPredictor


###############################################
# Config
###############################################

IMAGE_PATH = (
    "/home/chenkejing/database/test/data/images/train/image_batch1_1.jpg"
)


SAM3_MODEL = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/sam3.pt"
)


BPE_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/"
    "bpe_simple_vocab_16e6.txt.gz"
)


PROMPTS = [
    "carpet",
    "wire",
    "plastic bag"
]


DEVICE = "cuda"


SAVE_PATH = (
    "/tmp/sam3_grounding_result.jpg"
)



###############################################
# Visualization
###############################################

def draw_masks(
        image,
        masks,
        labels
):

    vis = image.copy()


    if masks is None:
        return vis


    for idx, mask in enumerate(masks):

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()


        mask = mask.astype(bool)


        color = np.random.randint(
            0,
            255,
            3
        )


        vis[mask] = (
            0.5 * vis[mask]
            +
            0.5 * color
        )


        ys, xs = np.where(mask)


        if len(xs) > 0:

            x1 = xs.min()
            y1 = ys.min()
            x2 = xs.max()
            y2 = ys.max()


            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                color.tolist(),
                2
            )


            cv2.putText(
                vis,
                labels[idx],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color.tolist(),
                2
            )


    return vis



###############################################
# Main Test
###############################################

def test_sam3_grounding():


    assert os.path.exists(
        IMAGE_PATH
    ), IMAGE_PATH


    assert os.path.exists(
        SAM3_MODEL
    ), SAM3_MODEL


    assert os.path.exists(
        BPE_PATH
    ), BPE_PATH



    print(
        "Loading SAM3..."
    )


    ###########################################
    # IMPORTANT:
    #
    # bpe_path is NOT YOLO argument
    #
    ###########################################

    predictor = SAM3SemanticPredictor(
        overrides={

            "model":
                SAM3_MODEL,

            "task":
                "segment",

            "device":
                DEVICE,

            "imgsz":
                672,

            "batch":
                1,
            "compile":
                False

        },

        bpe_path=
            BPE_PATH
    )


    print(
        "SAM3 loaded"
    )



    ###################################
    # Read image
    ###################################

    image = cv2.imread(
        IMAGE_PATH
    )

    assert image is not None



    ###################################
    # SAM3 inference
    ###################################

    results = []


    for prompt in PROMPTS:


        print(
            "\nPrompt:",
            prompt
        )


        result = predictor(
            source=IMAGE_PATH,
            text=prompt
        )


        results.append(
            result
        )


        print(
            result
        )



    ###################################
    # Parse result
    ###################################

    all_masks = []
    all_labels = []



    for prompt, result in zip(
        PROMPTS,
        results
    ):


        if len(result) == 0:
            continue


        r = result[0]


        ################################
        # masks
        ################################

        if r.masks is not None:

            masks = r.masks.data


            for mask in masks:

                all_masks.append(
                    mask
                )

                all_labels.append(
                    prompt
                )



        ################################
        # boxes
        ################################

        if r.boxes is not None:


            boxes = (
                r.boxes.xyxy
                .cpu()
                .numpy()
            )


            scores = (
                r.boxes.conf
                .cpu()
                .numpy()
            )


            for box, score in zip(
                boxes,
                scores
            ):

                print(
                    "class:",
                    prompt,
                    "box:",
                    box,
                    "score:",
                    float(score)
                )



    ###################################
    # visualize
    ###################################

    vis = draw_masks(
        image,
        all_masks,
        all_labels
    )


    cv2.imwrite(
        SAVE_PATH,
        vis
    )


    print(
        "\nSaved:",
        SAVE_PATH
    )



###############################################
# Entry
###############################################

if __name__ == "__main__":

    test_sam3_grounding()