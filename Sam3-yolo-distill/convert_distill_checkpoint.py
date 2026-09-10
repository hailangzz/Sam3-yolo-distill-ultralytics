# -*- coding: utf-8 -*-

"""
######################################################################
#
#  SAM3 -> YOLOv8-seg
#
#  Distillation Checkpoint Converter
#
######################################################################

Purpose
-------
Convert custom SAM3 -> YOLOv8-seg distillation checkpoint:

    checkpoints/best.pt

into a standard Ultralytics YOLOv8-seg model:

    checkpoints/distilled_yolov8n-seg.pt


IMPORTANT
---------
The current Student is a genuine 4-class YOLOv8n-seg model:

    0: carpet
    1: wire
    2: liquid
    3: plasticbag

Therefore the converter MUST reconstruct the Student architecture
with nc=4 before loading checkpoint["student"].

DO NOT do:

    YOLO("yolov8n-seg.pt")

because the original pretrained model is COCO:

    nc = 80

Instead:

    yolov8-seg.yaml
            |
            v
    SegmentationModel(nc=4)
            |
            v
    load checkpoint["student"]
            |
            v
    standard 4-class YOLOv8-seg
"""

import os
import sys

import torch

from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace


# ==============================================================
# Project configuration
# ==============================================================

PROJECT_DIR = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill"
)


# ==============================================================
# Input / output
# ==============================================================

DISTILL_CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best.pt"
)


OUTPUT_MODEL = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "distilled_yolov8n-seg.pt"
)


# ==============================================================
# Original YOLO pretrained model
# ==============================================================
#
# This model is ONLY used for checking that the original
# YOLOv8n-seg model exists.
#
# The converter DOES NOT load its 80-class architecture.
#
# The actual Student architecture is reconstructed from:
#
#     yolov8-seg.yaml
#
# with:
#
#     nc = 4
#
# =============================================================

YOLO_WEIGHTS_PATH = os.path.join(
    PROJECT_DIR,
    "yolov8n-seg.pt"
)


# ==============================================================
# Classes
# ==============================================================

CLASS_NAMES = {
    0: "carpet",
    1: "wire",
    2: "liquid",
    3: "plasticbag",
}

NUM_CLASSES = len(CLASS_NAMES)


# ==============================================================
# Device
# ==============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==============================================================
# Find Ultralytics YAML
# ==============================================================

import ultralytics


ULTRALYTICS_DIR = os.path.dirname(
    ultralytics.__file__
)


YOLO_YAML_PATH = os.path.join(
    ULTRALYTICS_DIR,
    "cfg",
    "models",
    "v8",
    "yolov8-seg.yaml"
)


# ==============================================================
# Utility
# ==============================================================

def print_separator():
    print("=" * 70)


# ==============================================================
# Find segmentation head
# ==============================================================

def find_segmentation_head(model):
    """
    Find YOLOv8 Segment head.

    In Ultralytics 8.3.237:

        model.nc

    may be None.

    Therefore the authoritative class count is:

        model.model[-1].nc
    """

    head = None

    for module in model.model:

        if (
            hasattr(module, "nc")
            and hasattr(module, "nl")
        ):
            head = module

    if head is None:

        raise RuntimeError(
            "Cannot find YOLO detection/segmentation head."
        )

    return head


# ==============================================================
# Build 4-class YOLOv8-seg model
# ==============================================================

def build_four_class_model():
    """
    Reconstruct the exact 4-class Student architecture.

    IMPORTANT:

    The checkpoint was trained using a Student with:

        nc = 4

    Therefore we must create:

        SegmentationModel(nc=4)

    before loading checkpoint["student"].
    """

    print_separator()

    print(
        "BUILD 4-CLASS YOLOv8n-seg MODEL"
    )

    print_separator()

    print(
        "YOLO YAML:"
    )

    print(
        "   ",
        YOLO_YAML_PATH
    )

    print(
        "YOLO pretrained:"
    )

    print(
        "   ",
        YOLO_WEIGHTS_PATH
    )

    print(
        "Number of classes:"
    )

    print(
        "   ",
        NUM_CLASSES
    )

    print(
        "Classes:"
    )

    print(
        "   ",
        CLASS_NAMES
    )

    # ==========================================================
    # Check files
    # ==========================================================

    if not os.path.exists(
        YOLO_YAML_PATH
    ):

        raise FileNotFoundError(
            "YOLO YAML not found:\n"
            f"{YOLO_YAML_PATH}"
        )

    if not os.path.exists(
        YOLO_WEIGHTS_PATH
    ):

        raise FileNotFoundError(
            "YOLO pretrained model not found:\n"
            f"{YOLO_WEIGHTS_PATH}"
        )

    # ==========================================================
    # Load YAML model
    # ==========================================================

    print()
    print(
        "Loading YOLO YAML..."
    )

    yaml_model = YOLO(
        YOLO_YAML_PATH
    )

    yaml_student = yaml_model.model

    # ==========================================================
    # Get YAML config
    # ==========================================================

    yaml_cfg = dict(
        yaml_student.yaml
    )

    # ==========================================================
    # Force nc = 4
    # ==========================================================

    yaml_cfg["nc"] = NUM_CLASSES

    print()
    print(
        "YAML configuration:"
    )

    print(
        "   nc =",
        yaml_cfg["nc"]
    )

    # ==========================================================
    # Rebuild 4-class SegmentationModel
    # ==========================================================

    from ultralytics.nn.tasks import SegmentationModel

    print()
    print(
        "Creating SegmentationModel(nc=4)..."
    )

    student = SegmentationModel(
        cfg=yaml_cfg,
        ch=3,
        nc=NUM_CLASSES,
        verbose=False
    )

    # ==========================================================
    # Find head
    # ==========================================================

    student_head = find_segmentation_head(
        student
    )

    print()
    print(
        "Student architecture:"
    )

    print(
        "   model type =",
        type(student)
    )

    print(
        "   head type =",
        type(student_head)
    )

    print(
        "   head nc =",
        student_head.nc
    )

    # ==========================================================
    # Verify nc
    # ==========================================================

    if student_head.nc != NUM_CLASSES:

        raise RuntimeError(
            "Failed to build 4-class YOLO model.\n"
            f"Expected head.nc = {NUM_CLASSES}\n"
            f"Actual head.nc = {student_head.nc}"
        )

    print()
    print(
        "4-class model architecture verified."
    )

    return student


# ==============================================================
# Load checkpoint
# ==============================================================

def load_distillation_checkpoint():

    print_separator()

    print(
        "LOAD DISTILLATION CHECKPOINT"
    )

    print_separator()

    print(
        "Checkpoint:"
    )

    print(
        "   ",
        DISTILL_CHECKPOINT
    )

    if not os.path.exists(
        DISTILL_CHECKPOINT
    ):

        raise FileNotFoundError(
            "Distillation checkpoint not found:\n"
            f"{DISTILL_CHECKPOINT}"
        )

    checkpoint = torch.load(
        DISTILL_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    print()
    print(
        "Checkpoint type:"
    )

    print(
        "   ",
        type(checkpoint)
    )

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Invalid distillation checkpoint. "
            "Expected a dictionary."
        )

    print()
    print(
        "Checkpoint keys:"
    )

    for key in checkpoint.keys():

        print(
            "   ",
            key
        )

    # ==========================================================
    # Print distillation metadata
    # ==========================================================

    print()
    print(
        "Distillation information:"
    )

    print(
        "   epoch:"
    )

    print(
        "      ",
        checkpoint.get(
            "epoch",
            None
        )
    )

    print(
        "   loss:"
    )

    print(
        "      ",
        checkpoint.get(
            "loss",
            None
        )
    )

    print(
        "   lambda_feature:"
    )

    print(
        "      ",
        checkpoint.get(
            "lambda_feature",
            None
        )
    )

    print(
        "   adapters:"
    )

    print(
        "      ",
        "present"
        if checkpoint.get("adapters") is not None
        else "not present"
    )

    print(
        "   optimizer:"
    )

    print(
        "      ",
        "present"
        if checkpoint.get("optimizer") is not None
        else "not present"
    )

    # ==========================================================
    # Student state
    # ==========================================================

    if "student" not in checkpoint:

        raise RuntimeError(
            "Distillation checkpoint does not contain "
            "`student` state_dict."
        )

    student_state = checkpoint["student"]

    if not isinstance(
        student_state,
        dict
    ):

        raise RuntimeError(
            "checkpoint['student'] is not a state_dict."
        )

    print()
    print(
        "Student state_dict parameters:"
    )

    print(
        "   ",
        len(student_state)
    )

    return checkpoint


# ==============================================================
# Load Student state_dict
# ==============================================================

def load_student_weights(
    student,
    checkpoint
):
    """
    Load the 4-class distillation Student weights.

    Since the model architecture has been reconstructed with
    exactly nc=4, all Student parameters should match.

    We perform a detailed shape check before loading.
    """

    print_separator()

    print(
        "LOAD DISTILLED STUDENT WEIGHTS"
    )

    print_separator()

    student_state = checkpoint[
        "student"
    ]

    current_state = student.state_dict()

    compatible_state = {}

    skipped = []

    # ==========================================================
    # Compare every checkpoint parameter
    # ==========================================================

    for key, value in student_state.items():

        # ------------------------------------------------------
        # Key missing
        # ------------------------------------------------------

        if key not in current_state:

            skipped.append(
                (
                    key,
                    "key_not_found"
                )
            )

            continue

        # ------------------------------------------------------
        # Shape mismatch
        # ------------------------------------------------------

        if (
            current_state[key].shape
            != value.shape
        ):

            skipped.append(
                (
                    key,
                    (
                        "shape mismatch: "
                        f"checkpoint "
                        f"{tuple(value.shape)} "
                        f"-> model "
                        f"{tuple(current_state[key].shape)}"
                    )
                )
            )

            continue

        # ------------------------------------------------------
        # Compatible
        # ------------------------------------------------------

        compatible_state[key] = value

    # ==========================================================
    # Print statistics
    # ==========================================================

    print()

    print(
        "Checkpoint parameters:"
    )

    print(
        "   ",
        len(student_state)
    )

    print(
        "Compatible parameters:"
    )

    print(
        "   ",
        len(compatible_state)
    )

    print(
        "Skipped parameters:"
    )

    print(
        "   ",
        len(skipped)
    )

    # ==========================================================
    # Print mismatches
    # ==========================================================

    if len(skipped) > 0:

        print()
        print(
            "Skipped / incompatible parameters:"
        )

        for key, reason in skipped:

            print(
                "   ",
                key,
                "->",
                reason
            )

    # ==========================================================
    # Correct architecture should have ZERO mismatches
    # ==========================================================

    if len(skipped) > 0:

        raise RuntimeError(
            "\n"
            "Student checkpoint and model architecture "
            "do not match.\n\n"
            f"Checkpoint parameters: {len(student_state)}\n"
            f"Compatible parameters: {len(compatible_state)}\n"
            f"Skipped parameters: {len(skipped)}\n\n"
            "This usually means the Student architecture "
            "used during training is different from the "
            "architecture used during conversion."
        )

    # ==========================================================
    # Load
    # ==========================================================

    print()
    print(
        "Loading state_dict..."
    )

    result = student.load_state_dict(
        compatible_state,
        strict=True
    )

    print()
    print(
        "state_dict loaded successfully."
    )

    print(
        "   missing keys =",
        len(result.missing_keys)
    )

    print(
        "   unexpected keys =",
        len(result.unexpected_keys)
    )

    if len(result.missing_keys) > 0:

        raise RuntimeError(
            "Unexpected missing keys after loading:\n"
            + "\n".join(
                result.missing_keys
            )
        )

    if len(result.unexpected_keys) > 0:

        raise RuntimeError(
            "Unexpected keys after loading:\n"
            + "\n".join(
                result.unexpected_keys
            )
        )


# ==============================================================
# Configure standard Ultralytics model
# ==============================================================

def configure_model(student):
    """
    Configure the reconstructed Student so it can be saved and
    reloaded by standard Ultralytics YOLO().
    """

    print_separator()

    print(
        "CONFIGURE STANDARD ULTRALYTICS MODEL"
    )

    print_separator()

    # ==========================================================
    # Class names
    # ==========================================================
    #
    # IMPORTANT:
    #
    # Do NOT do:
    #
    #     model.names = CLASS_NAMES
    #
    # YOLO.names is a read-only property in this version.
    #
    # names should be attached to the underlying model.
    #
    # ==========================================================

    student.names = CLASS_NAMES

    # ==========================================================
    # Ensure args exists
    # ==========================================================

    model_args = getattr(
        student,
        "args",
        None
    )

    if model_args is None:

        model_args = {}

    elif isinstance(
        model_args,
        IterableSimpleNamespace
    ):

        model_args = vars(
            model_args
        )

    elif isinstance(
        model_args,
        dict
    ):

        model_args = dict(
            model_args
        )

    else:

        model_args = vars(
            model_args
        )

    # ==========================================================
    # Model / training metadata
    # ==========================================================

    model_args["nc"] = NUM_CLASSES

    model_args["task"] = "segment"

    model_args["overlap_mask"] = True

    model_args["mask_ratio"] = 4

    model_args["box"] = 7.5

    model_args["cls"] = 0.5

    model_args["dfl"] = 1.5

    student.args = IterableSimpleNamespace(
        **model_args
    )

    # ==========================================================
    # Set task on underlying model
    # ==========================================================

    student.task = "segment"

    # ==========================================================
    # Verify
    # ==========================================================

    head = find_segmentation_head(
        student
    )

    print()
    print(
        "Model metadata:"
    )

    print(
        "   head nc =",
        head.nc
    )

    print(
        "   args.nc =",
        student.args.nc
    )

    print(
        "   names =",
        student.names
    )

    print(
        "   task =",
        student.task
    )

    # ==========================================================
    # Hard verification
    # ==========================================================

    if head.nc != NUM_CLASSES:

        raise RuntimeError(
            "Model head class count is incorrect:\n"
            f"{head.nc}"
        )

    if student.args.nc != NUM_CLASSES:

        raise RuntimeError(
            "Model args.nc is incorrect:\n"
            f"{student.args.nc}"
        )

    if len(student.names) != NUM_CLASSES:

        raise RuntimeError(
            "Model class names count is incorrect:\n"
            f"{len(student.names)}"
        )

    expected_names = list(
        CLASS_NAMES.values()
    )

    actual_names = [
        student.names[i]
        for i in range(NUM_CLASSES)
    ]

    if actual_names != expected_names:

        raise RuntimeError(
            "Model class names are incorrect.\n"
            f"Expected: {expected_names}\n"
            f"Actual:   {actual_names}"
        )


# ==============================================================
# Save standard YOLO model
# ==============================================================

def save_standard_model(student):

    print_separator()

    print(
        "SAVE STANDARD YOLO MODEL"
    )

    print_separator()

    # ==========================================================
    # Move to CPU before saving
    # ==========================================================

    student = student.cpu()

    student.eval()

    # ==========================================================
    # IMPORTANT
    #
    # The names are already stored in:
    #
    #     student.names
    #
    # DO NOT assign:
    #
    #     model.names = CLASS_NAMES
    #
    # because YOLO.names is read-only.
    # ==========================================================

    print()
    print(
        "Student before wrapping:"
    )

    head = find_segmentation_head(
        student
    )

    print(
        "   head nc =",
        head.nc
    )

    print(
        "   args.nc =",
        student.args.nc
    )

    print(
        "   names =",
        student.names
    )

    print(
        "   task =",
        getattr(
            student,
            "task",
            None
        )
    )

    # ==========================================================
    # Wrap in Ultralytics YOLO object
    # ==========================================================

    print()
    print(
        "Creating YOLO wrapper..."
    )

    model = YOLO(
        YOLO_YAML_PATH
    )

    # ==========================================================
    # Replace underlying model
    # ==========================================================

    model.model = student

    # ==========================================================
    # Set task
    # ==========================================================

    model.task = "segment"

    # ==========================================================
    # IMPORTANT:
    #
    # DO NOT:
    #
    #     model.names = CLASS_NAMES
    #
    # YOLO.names is read-only.
    #
    # The property should obtain names from:
    #
    #     model.model.names
    #
    # ==========================================================

    print()
    print(
        "YOLO wrapper:"
    )

    print(
        "   task =",
        model.task
    )

    print(
        "   names =",
        model.names
    )

    # ==========================================================
    # Save
    # ==========================================================

    print()
    print(
        "Saving:"
    )

    print(
        "   ",
        OUTPUT_MODEL
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_MODEL
        ),
        exist_ok=True
    )

    model.save(
        OUTPUT_MODEL
    )

    print()
    print(
        "Model saved successfully."
    )

    return model


# ==============================================================
# Verify saved model
# ==============================================================

def verify_saved_model():

    print_separator()

    print(
        "VERIFY SAVED MODEL"
    )

    print_separator()

    print()
    print(
        "Reloading:"
    )

    print(
        "   ",
        OUTPUT_MODEL
    )

    if not os.path.exists(
        OUTPUT_MODEL
    ):

        raise FileNotFoundError(
            "Output model was not created:\n"
            f"{OUTPUT_MODEL}"
        )

    # ==========================================================
    # Reload
    # ==========================================================

    reloaded = YOLO(
        OUTPUT_MODEL
    )

    print()
    print(
        "Reload successful."
    )

    # ==========================================================
    # Basic information
    # ==========================================================

    print()
    print(
        "Reloaded model:"
    )

    print(
        "   task =",
        reloaded.task
    )

    print(
        "   model type =",
        type(reloaded.model)
    )

    # ==========================================================
    # Head
    # ==========================================================

    head = find_segmentation_head(
        reloaded.model
    )

    print(
        "   head type =",
        type(head)
    )

    print(
        "   head nc =",
        head.nc
    )

    # ==========================================================
    # Names
    # ==========================================================

    print(
        "   names =",
        reloaded.names
    )

    # ==========================================================
    # Underlying model names
    # ==========================================================

    print(
        "   model.names =",
        getattr(
            reloaded.model,
            "names",
            None
        )
    )

    # ==========================================================
    # args
    # ==========================================================

    print(
        "   args.nc =",
        getattr(
            getattr(
                reloaded.model,
                "args",
                None
            ),
            "nc",
            None
        )
    )

    # ==========================================================
    # Parameter count
    # ==========================================================

    parameters = sum(
        p.numel()
        for p in reloaded.model.parameters()
    )

    print(
        "   parameters =",
        parameters
    )

    # ==========================================================
    # Hard verification
    # ==========================================================

    if reloaded.task != "segment":

        raise RuntimeError(
            "VERIFICATION FAILED:\n"
            f"Expected task='segment', "
            f"actual task='{reloaded.task}'"
        )

    if head.nc != NUM_CLASSES:

        raise RuntimeError(
            "VERIFICATION FAILED:\n"
            f"Expected nc={NUM_CLASSES}, "
            f"actual nc={head.nc}"
        )

    if not hasattr(
        reloaded.model,
        "names"
    ):

        raise RuntimeError(
            "VERIFICATION FAILED:\n"
            "Reloaded model has no `names` attribute."
        )

    if len(reloaded.names) != NUM_CLASSES:

        raise RuntimeError(
            "VERIFICATION FAILED:\n"
            f"Expected {NUM_CLASSES} names, "
            f"actual {len(reloaded.names)}"
        )

    expected_names = list(
        CLASS_NAMES.values()
    )

    actual_names = [
        reloaded.names[i]
        for i in range(NUM_CLASSES)
    ]

    if actual_names != expected_names:

        raise RuntimeError(
            "VERIFICATION FAILED:\n"
            f"Expected names={expected_names}\n"
            f"Actual names={actual_names}"
        )

    # ==========================================================
    # Verify args.nc if available
    # ==========================================================

    reloaded_args = getattr(
        reloaded.model,
        "args",
        None
    )

    if reloaded_args is not None:

        reloaded_nc = getattr(
            reloaded_args,
            "nc",
            None
        )

        if reloaded_nc is not None:

            if reloaded_nc != NUM_CLASSES:

                raise RuntimeError(
                    "VERIFICATION FAILED:\n"
                    f"Expected args.nc={NUM_CLASSES}, "
                    f"actual args.nc={reloaded_nc}"
                )

    # ==========================================================
    # Success
    # ==========================================================

    print()
    print_separator()

    print(
        "MODEL VERIFICATION SUCCESS"
    )

    print_separator()

    print()
    print(
        "Final model:"
    )

    print(
        "   task =",
        reloaded.task
    )

    print(
        "   nc =",
        head.nc
    )

    print(
        "   classes =",
        reloaded.names
    )

    print(
        "   parameters =",
        parameters
    )

    print()

    return reloaded


# ==============================================================
# Main
# ==============================================================

def main():

    print()
    print("#" * 70)
    print("#")
    print("#  SAM3 -> YOLOv8-seg")
    print("#")
    print("#  Distillation Checkpoint Converter")
    print("#")
    print("#" * 70)

    # ==========================================================
    # Environment
    # ==========================================================

    print()
    print(
        "Project:"
    )

    print(
        "   ",
        PROJECT_DIR
    )

    print()
    print(
        "Python:"
    )

    print(
        "   ",
        sys.version
    )

    print()
    print(
        "PyTorch:"
    )

    print(
        "   ",
        torch.__version__
    )

    print()
    print(
        "CUDA available:"
    )

    print(
        "   ",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print()
        print(
            "CUDA device:"
        )

        print(
            "   ",
            torch.cuda.get_device_name(0)
        )

    print()

    # ==========================================================
    # Load checkpoint
    # ==========================================================

    checkpoint = load_distillation_checkpoint()

    # ==========================================================
    # Build exact 4-class architecture
    # ==========================================================

    student = build_four_class_model()

    # ==========================================================
    # Load distilled weights
    # ==========================================================

    load_student_weights(
        student,
        checkpoint
    )

    # ==========================================================
    # Configure model
    # ==========================================================

    configure_model(
        student
    )

    # ==========================================================
    # Save
    # ==========================================================

    save_standard_model(
        student
    )

    # ==========================================================
    # Reload and verify
    # ==========================================================

    verify_saved_model()

    # ==========================================================
    # Finished
    # ==========================================================

    print()
    print("#" * 70)
    print("#")
    print("#  CONVERSION COMPLETE")
    print("#")
    print("#" * 70)

    print()
    print(
        "Output:"
    )

    print(
        "   ",
        OUTPUT_MODEL
    )

    print()
    print(
        "Classes:"
    )

    for class_id, name in CLASS_NAMES.items():

        print(
            f"   {class_id}: {name}"
        )

    print()
    print(
        "This model is now a standard Ultralytics"
    )

    print(
        "YOLOv8n-seg 4-class model."
    )

    print()
    print(
        "It can be used for:"
    )

    print(
        "   1. Normal YOLOv8-seg inference"
    )

    print(
        "   2. Validation"
    )

    print(
        "   3. Further fine-tuning"
    )

    print(
        "   4. Export to deployment formats"
    )


# ==============================================================
# Entry
# ==============================================================

if __name__ == "__main__":

    main()

