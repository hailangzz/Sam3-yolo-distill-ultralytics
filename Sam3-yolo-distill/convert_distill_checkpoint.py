# -*- coding: utf-8 -*-

"""
Convert SAM3 -> YOLOv8-seg distillation checkpoint
to a standard Ultralytics YOLO checkpoint.

Input:
    checkpoints/best.pt

The distillation checkpoint contains:
    {
        "epoch": ...,
        "loss": ...,
        "student": ...,
        "adapters": ...,
        "optimizer": ...,
        "scaler": ...,
        "lambda_feature": ...
    }

Only checkpoint["student"] is used.

The final model is a normal YOLOv8-seg model and can be used for:

    model.predict(...)
    model.val(...)
    model.train(...)
    model.export(...)

SAM3 / Adapter / FeatureLoss are NOT included in the final model.
"""

import os
import sys
import torch

from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

# Current project directory
PROJECT_DIR = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill"
)

# Original YOLO architecture / pretrained model
YOLO_PATH = os.path.join(
    PROJECT_DIR,
    "yolov8n-seg.pt"
)

# Distillation checkpoint
DISTILL_CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best.pt"
)

# Output standard YOLO checkpoint
OUTPUT_MODEL = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "distilled_yolov8n-seg.pt"
)


# ============================================================
# Utility
# ============================================================

def print_model_info(model):
    """Print basic model information."""

    print()
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    print("Model type:")
    print("   ", type(model.model))

    print("Model task:")
    print("   ", getattr(model, "task", None))

    print("Number of parameters:")
    print(
        "   ",
        sum(p.numel() for p in model.model.parameters())
    )

    print("Number of trainable parameters:")
    print(
        "   ",
        sum(
            p.numel()
            for p in model.model.parameters()
            if p.requires_grad
        )
    )

    print("=" * 70)


# ============================================================
# Load distillation checkpoint
# ============================================================

def load_distillation_checkpoint(path):
    print()
    print("=" * 70)
    print("LOAD DISTILLATION CHECKPOINT")
    print("=" * 70)

    print("Checkpoint:")
    print("   ", path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Distillation checkpoint does not exist:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    print()
    print("Checkpoint type:")
    print("   ", type(checkpoint))

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Invalid distillation checkpoint: "
            "checkpoint is not a dictionary."
        )

    print()
    print("Checkpoint keys:")

    for key in checkpoint.keys():
        print("   ", key)

    # --------------------------------------------------------
    # Check student weights
    # --------------------------------------------------------

    if "student" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain 'student'.\n"
            f"Available keys: {list(checkpoint.keys())}"
        )

    student_state_dict = checkpoint["student"]

    if not isinstance(student_state_dict, dict):
        raise RuntimeError(
            "checkpoint['student'] is not a state_dict dictionary."
        )

    print()
    print("Distillation information:")

    if "epoch" in checkpoint:
        print("   epoch:")
        print("      ", checkpoint["epoch"])

    if "loss" in checkpoint:
        print("   loss:")
        print("      ", checkpoint["loss"])

    if "lambda_feature" in checkpoint:
        print("   lambda_feature:")
        print("      ", checkpoint["lambda_feature"])

    if "adapters" in checkpoint:
        print("   adapters:")
        print("      present")

    if "optimizer" in checkpoint:
        print("   optimizer:")
        print("      present")

    print()
    print("Student state_dict parameters:")
    print("   ", len(student_state_dict))

    return checkpoint, student_state_dict


# ============================================================
# Build original YOLO model
# ============================================================

def build_yolo_model():
    print()
    print("=" * 70)
    print("LOAD ORIGINAL YOLO MODEL")
    print("=" * 70)

    print("YOLO architecture:")
    print("   ", YOLO_PATH)

    if not os.path.exists(YOLO_PATH):
        raise FileNotFoundError(
            f"YOLO model does not exist:\n{YOLO_PATH}"
        )

    # Important:
    #
    # This is exactly the same construction used in
    # train_distill.py:
    #
    #     student = YOLO(YOLO_PATH).model
    #
    model = YOLO(YOLO_PATH)

    student = model.model

    print()
    print("YOLO model loaded successfully.")

    print("Model type:")
    print("   ", type(student))

    print("Task:")
    print("   ", getattr(model, "task", None))

    print("Number of parameters:")
    print(
        "   ",
        sum(p.numel() for p in student.parameters())
    )

    return model, student


# ============================================================
# Load distilled weights
# ============================================================

def load_student_weights(student, student_state_dict):
    print()
    print("=" * 70)
    print("LOAD DISTILLED STUDENT WEIGHTS")
    print("=" * 70)

    print("Loading checkpoint['student'] ...")

    # strict=True is intentional.
    #
    # Your distillation code does not modify YOLO architecture.
    # Therefore the state_dict should exactly match the original
    # YOLOv8n-seg model.
    #
    # If strict=True fails, it means something is inconsistent
    # between the model used during distillation and YOLO_PATH.
    result = student.load_state_dict(
        student_state_dict,
        strict=True
    )

    print()
    print("Student weights loaded successfully.")

    print("Missing keys:")
    print("   ", result.missing_keys)

    print("Unexpected keys:")
    print("   ", result.unexpected_keys)

    if result.missing_keys:
        raise RuntimeError(
            "Missing keys detected when loading student weights."
        )

    if result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys detected when loading student weights."
        )


# ============================================================
# Save standard Ultralytics model
# ============================================================

def save_standard_yolo_model(model, output_path):
    print()
    print("=" * 70)
    print("SAVE STANDARD ULTRALYTICS MODEL")
    print("=" * 70)

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    print("Output:")
    print("   ", output_path)

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # model is the high-level Ultralytics YOLO object.
    #
    # model.save() creates a normal Ultralytics checkpoint,
    # rather than our custom distillation checkpoint.
    #
    model.save(output_path)

    if not os.path.exists(output_path):
        raise RuntimeError(
            "Model save failed: output file was not created."
        )

    file_size_mb = (
        os.path.getsize(output_path)
        / 1024
        / 1024
    )

    print()
    print("Model saved successfully.")

    print("File:")
    print("   ", output_path)

    print("Size:")
    print(
        "   {:.2f} MB".format(file_size_mb)
    )


# ============================================================
# Verify standard checkpoint
# ============================================================

def verify_standard_model(output_path):
    print()
    print("=" * 70)
    print("VERIFY STANDARD YOLO CHECKPOINT")
    print("=" * 70)

    print("Reload:")
    print("   ", output_path)

    # This is the most important verification.
    #
    # If this succeeds, the converted model can be loaded by
    # normal Ultralytics code.
    verify_model = YOLO(output_path)

    print()
    print("SUCCESS:")
    print("   YOLO checkpoint can be loaded normally.")

    print()
    print("Task:")
    print("   ", verify_model.task)

    print("Model type:")
    print("   ", type(verify_model.model))

    print("Number of parameters:")
    print(
        "   ",
        sum(
            p.numel()
            for p in verify_model.model.parameters()
        )
    )

    print_model_info(verify_model)

    return verify_model


# ============================================================
# Compare weights
# ============================================================

def verify_weights(original_student, converted_model):
    print()
    print("=" * 70)
    print("VERIFY WEIGHTS")
    print("=" * 70)

    original_state = original_student.state_dict()
    converted_state = converted_model.model.state_dict()

    print("Original state_dict keys:")
    print("   ", len(original_state))

    print("Converted state_dict keys:")
    print("   ", len(converted_state))

    if set(original_state.keys()) != set(converted_state.keys()):
        print()
        print("WARNING:")
        print("State dict keys are different.")

        missing = set(original_state.keys()) - set(
            converted_state.keys()
        )

        unexpected = set(converted_state.keys()) - set(
            original_state.keys()
        )

        if missing:
            print()
            print("Missing:")
            for key in sorted(missing):
                print("   ", key)

        if unexpected:
            print()
            print("Unexpected:")
            for key in sorted(unexpected):
                print("   ", key)

        raise RuntimeError(
            "Converted model state_dict structure differs "
            "from the original student model."
        )

    max_difference = 0.0
    different_keys = []

    for key in original_state.keys():

        a = original_state[key].detach().cpu()
        b = converted_state[key].detach().cpu()

        if a.shape != b.shape:
            raise RuntimeError(
                f"Shape mismatch for key: {key}\n"
                f"Original: {a.shape}\n"
                f"Converted: {b.shape}"
            )

        # Convert to float32 for comparison.
        diff = (
            a.float() - b.float()
        ).abs().max().item()

        max_difference = max(
            max_difference,
            diff
        )

        if diff != 0:
            different_keys.append(
                (key, diff)
            )

    print()
    print("Maximum weight difference:")
    print("   ", max_difference)

    if max_difference == 0:
        print()
        print("SUCCESS:")
        print("   Converted model weights are EXACTLY")
        print("   the same as checkpoint['student'].")
    else:
        print()
        print("WARNING:")
        print("   Converted model contains numerical differences.")

        print()
        print("Number of different tensors:")
        print("   ", len(different_keys))

    return max_difference


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("#" * 70)
    print("#")
    print("#  SAM3 -> YOLOv8-seg")
    print("#")
    print("#  Distillation Checkpoint Converter")
    print("#")
    print("#" * 70)

    print()
    print("Project:")
    print("   ", PROJECT_DIR)

    print()
    print("Python:")
    print("   ", sys.version)

    print()
    print("PyTorch:")
    print("   ", torch.__version__)

    print()
    print("CUDA available:")
    print("   ", torch.cuda.is_available())

    if torch.cuda.is_available():
        print()
        print("CUDA device:")
        print(
            "   ",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # 1. Load distillation checkpoint
    # --------------------------------------------------------

    checkpoint, student_state_dict = (
        load_distillation_checkpoint(
            DISTILL_CHECKPOINT
        )
    )

    # --------------------------------------------------------
    # 2. Build exactly the same YOLO student architecture
    # --------------------------------------------------------

    model, student = build_yolo_model()

    # --------------------------------------------------------
    # 3. Load distilled weights
    # --------------------------------------------------------

    load_student_weights(
        student,
        student_state_dict
    )

    # --------------------------------------------------------
    # 4. Save as standard Ultralytics checkpoint
    # --------------------------------------------------------

    save_standard_yolo_model(
        model,
        OUTPUT_MODEL
    )

    # --------------------------------------------------------
    # 5. Reload checkpoint
    # --------------------------------------------------------

    converted_model = verify_standard_model(
        OUTPUT_MODEL
    )

    # --------------------------------------------------------
    # 6. Verify weights
    # --------------------------------------------------------

    verify_weights(
        student,
        converted_model
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("#" * 70)
    print("#")
    print("#  CONVERSION SUCCESS")
    print("#")
    print("#" * 70)

    print()
    print("Input distillation checkpoint:")
    print("   ", DISTILL_CHECKPOINT)

    print()
    print("Output standard YOLO checkpoint:")
    print("   ", OUTPUT_MODEL)

    print()
    print("You can now use:")

    print()
    print('   from ultralytics import YOLO')
    print()
    print(
        f'   model = YOLO("{OUTPUT_MODEL}")'
    )

    print()
    print("Supported operations:")

    print("   model.predict(...)")
    print("   model.val(...)")
    print("   model.train(...)")
    print("   model.export(...)")

    print()
    print("SAM3 / Adapter / FeatureLoss are not required")
    print("for normal inference, fine-tuning, validation,")
    print("or deployment.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()