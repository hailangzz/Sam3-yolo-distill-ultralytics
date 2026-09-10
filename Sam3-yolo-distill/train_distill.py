
# -*- coding: utf-8 -*-

"""
SAM3 -> YOLOv8-seg Feature Distillation

Student:
    YOLOv8n-seg
    nc = 4

Classes:
    0: carpet
    1: wire
    2: liquid
    3: plasticbag

Important:
    Student architecture is explicitly built as nc=4.

    We DO NOT directly use the original COCO 80-class
    YOLOv8n-seg model as the final Student.

    Instead:

        yolov8n-seg.yaml
                |
                v
        build YOLOv8n-seg
                |
                v
             nc = 4
                |
                v
        load compatible weights
        from yolov8n-seg.pt
                |
                v
        4-class Student

No training loop here.
Only build trainer.
"""

import os

import torch

from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace

from teacher.sam3_teacher import SAM3Teacher
from modules.adapters import FeatureAdapter
from losses.feature_loss import FeatureLoss
from distill_trainer import DistillTrainer
from hooks.yolo_hook import YOLOFeatureHook


# =====================================================
# config
# =====================================================

DEVICE = "cuda"


PROJECT_DIR = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill"
)


# =====================================================
# Original YOLO pretrained weights
# =====================================================

YOLO_WEIGHTS_PATH = os.path.join(
    PROJECT_DIR,
    "yolov8n-seg.pt"
)


# =====================================================
# YOLO architecture YAML
# =====================================================

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


# =====================================================
# SAM3
# =====================================================

SAM3_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "sam3.pt"
)


BPE_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "bpe_simple_vocab_16e6.txt.gz"
)


# =====================================================
# Classes
# =====================================================

CLASS_NAMES = {
    0: "carpet",
    1: "wire",
    2: "liquid",
    3: "plasticbag",
}

NUM_CLASSES = len(CLASS_NAMES)


# =====================================================
# helper
# =====================================================

def find_segmentation_head(student):
    """
    Find YOLOv8 Segment / Detect head.

    In Ultralytics 8.3.237:

        student.nc

    may be None.

    Therefore the authoritative class count is:

        student.model[-1].nc

    or the detected head's .nc.
    """

    segmentation_head = None

    for module in student.model:

        # YOLOv8 Segment inherits from Detect
        # and contains nc / nl.
        if (
            hasattr(module, "nc")
            and hasattr(module, "nl")
        ):
            segmentation_head = module

    if segmentation_head is None:

        raise RuntimeError(
            "Cannot find YOLO detection/segmentation head."
        )

    return segmentation_head


# =====================================================
# Build Student
# =====================================================

def build_four_class_student():
    """
    Build a genuine YOLOv8n-seg model with nc=4.

    Steps:

        1. Load YOLOv8n-seg YAML
        2. Rebuild SegmentationModel with nc=4
        3. Load compatible parameters from COCO pretrained model
        4. Skip incompatible 80-class classification parameters
        5. Set class names
        6. Configure segmentation loss
        7. Verify actual Segment head nc == 4
    """

    print()
    print("================================================")
    print("BUILD YOLOv8n-seg STUDENT")
    print("================================================")

    print(
        "YOLO YAML:",
        YOLO_YAML_PATH
    )

    print(
        "YOLO pretrained:",
        YOLO_WEIGHTS_PATH
    )

    print(
        "Number of classes:",
        NUM_CLASSES
    )

    print(
        "Classes:",
        CLASS_NAMES
    )

    # =================================================
    # Check files
    # =================================================

    if not os.path.exists(YOLO_WEIGHTS_PATH):

        raise FileNotFoundError(
            "YOLO pretrained model not found:\n"
            f"{YOLO_WEIGHTS_PATH}"
        )

    if not os.path.exists(YOLO_YAML_PATH):

        raise FileNotFoundError(
            "YOLO YAML not found:\n"
            f"{YOLO_YAML_PATH}\n\n"
            "Please check your installed Ultralytics package."
        )

    # =================================================
    # Load YAML once
    # =================================================

    print()
    print("Loading YOLO YAML...")

    original_model = YOLO(
        YOLO_YAML_PATH
    )

    original_student = original_model.model

    print(
        "Initial model type:",
        type(original_student)
    )

    print(
        "Initial model nc:",
        getattr(
            original_student,
            "nc",
            None
        )
    )

    # =================================================
    # Inspect original head
    # =================================================

    original_head = find_segmentation_head(
        original_student
    )

    print()
    print("Initial YOLO head:")
    print(
        "   type =",
        type(original_head)
    )

    print(
        "   nc =",
        getattr(
            original_head,
            "nc",
            None
        )
    )

    # =================================================
    # Get YAML configuration
    # =================================================

    yaml_cfg = dict(
        original_student.yaml
    )

    # Force 4 classes.
    yaml_cfg["nc"] = NUM_CLASSES

    print()
    print("Student YAML configuration:")
    print(
        "   nc =",
        yaml_cfg["nc"]
    )

    # =================================================
    # Rebuild genuine 4-class SegmentationModel
    # =================================================

    print()
    print("================================================")
    print("REBUILD 4-CLASS STUDENT")
    print("================================================")

    from ultralytics.nn.tasks import SegmentationModel

    student = SegmentationModel(
        cfg=yaml_cfg,
        ch=3,
        nc=NUM_CLASSES,
        verbose=False
    )

    # =================================================
    # Find actual Student head
    # =================================================

    student_head = find_segmentation_head(
        student
    )

    print()
    print("New Student architecture:")
    print(
        "   type =",
        type(student)
    )

    print(
        "   head type =",
        type(student_head)
    )

    print(
        "   head nc =",
        getattr(
            student_head,
            "nc",
            None
        )
    )

    # =================================================
    # IMPORTANT
    #
    # In Ultralytics 8.3.237:
    #
    #     student.nc
    #
    # may be None.
    #
    # The real class count is:
    #
    #     student.model[-1].nc
    #
    # Therefore we validate the actual head.
    # =================================================

    if getattr(
        student_head,
        "nc",
        None
    ) != NUM_CLASSES:

        raise RuntimeError(
            "Failed to build 4-class Student.\n"
            f"Expected head.nc = {NUM_CLASSES}\n"
            f"Actual head.nc = "
            f"{getattr(student_head, 'nc', None)}"
        )

    print()
    print(
        "4-class Student architecture created successfully."
    )

    # =================================================
    # Load original COCO pretrained model
    # =================================================

    print()
    print("================================================")
    print("LOAD COCO PRETRAINED MODEL")
    print("================================================")

    pretrained_model = YOLO(
        YOLO_WEIGHTS_PATH
    ).model

    pretrained_head = find_segmentation_head(
        pretrained_model
    )

    print(
        "Pretrained model type:",
        type(pretrained_model)
    )

    print(
        "Pretrained head type:",
        type(pretrained_head)
    )

    print(
        "Pretrained head nc:",
        getattr(
            pretrained_head,
            "nc",
            None
        )
    )

    # =================================================
    # Transfer compatible parameters
    # =================================================

    print()
    print("================================================")
    print("TRANSFER COMPATIBLE PRETRAINED WEIGHTS")
    print("================================================")

    student_state = student.state_dict()

    pretrained_state = pretrained_model.state_dict()

    compatible_state = {}

    skipped = []

    for key, value in pretrained_state.items():

        # -------------------------------------------------
        # Key does not exist
        # -------------------------------------------------

        if key not in student_state:

            skipped.append(
                (
                    key,
                    "key_not_found"
                )
            )

            continue

        # -------------------------------------------------
        # Shape mismatch
        # -------------------------------------------------

        if student_state[key].shape != value.shape:

            skipped.append(
                (
                    key,
                    "shape mismatch: "
                    f"{tuple(value.shape)} -> "
                    f"{tuple(student_state[key].shape)}"
                )
            )

            continue

        # -------------------------------------------------
        # Compatible
        # -------------------------------------------------

        compatible_state[key] = value

    # =================================================
    # Load compatible parameters
    # =================================================

    result = student.load_state_dict(
        compatible_state,
        strict=False
    )

    print()
    print("Pretrained weight transfer result:")

    print(
        "   compatible parameters =",
        len(compatible_state)
    )

    print(
        "   skipped parameters =",
        len(skipped)
    )

    print(
        "   missing keys =",
        len(result.missing_keys)
    )

    print(
        "   unexpected keys =",
        len(result.unexpected_keys)
    )

    # =================================================
    # Print skipped parameters
    # =================================================

    print()
    print("Skipped pretrained parameters:")

    if len(skipped) == 0:

        print(
            "   None"
        )

    else:

        for key, reason in skipped:

            print(
                "   ",
                key,
                "->",
                reason
            )

    # =================================================
    # Verify final Student head
    # =================================================

    print()
    print("================================================")
    print("VERIFY 4-CLASS STUDENT")
    print("================================================")

    student_head = find_segmentation_head(
        student
    )

    student_head_nc = getattr(
        student_head,
        "nc",
        None
    )

    print(
        "Student type:",
        type(student)
    )

    print(
        "Student head:",
        type(student_head)
    )

    print(
        "Student head nc:",
        student_head_nc
    )

    print(
        "Expected nc:",
        NUM_CLASSES
    )

    # -------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT do:
    #
    #     student.nc
    #
    # because Ultralytics 8.3.237 may return None.
    #
    # We only trust:
    #
    #     student.model[-1].nc
    # -------------------------------------------------

    if student_head_nc != NUM_CLASSES:

        raise RuntimeError(
            "Student segmentation head nc is incorrect.\n"
            f"Expected: {NUM_CLASSES}\n"
            f"Actual: {student_head_nc}"
        )

    print()
    print(
        "4-class Student verification SUCCESS"
    )

    # =================================================
    # Set class names
    # =================================================

    student.names = CLASS_NAMES

    # =================================================
    # Normalize args
    # =================================================

    if isinstance(
        getattr(student, "args", None),
        dict
    ):

        student.args = IterableSimpleNamespace(
            **student.args
        )

    # =================================================
    # Set class names
    # =================================================

    student.names = CLASS_NAMES

    # =================================================
    # Ensure model args exists
    # =================================================
    #
    # Directly constructed SegmentationModel in
    # Ultralytics 8.3.237 may NOT have `args`.
    #
    # Therefore:
    #
    #     student.args
    #
    # cannot be assumed to exist.
    #
    # We create it if necessary.
    # =================================================

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

        model_args = vars(model_args)

    elif not isinstance(
            model_args,
            dict
    ):

        model_args = vars(model_args)

    # =================================================
    # Set training arguments
    # =================================================

    model_args["nc"] = NUM_CLASSES

    model_args["overlap_mask"] = True

    model_args["mask_ratio"] = 4

    model_args["box"] = 7.5

    model_args["cls"] = 0.5

    model_args["dfl"] = 1.5

    # =================================================
    # Convert to IterableSimpleNamespace
    # =================================================

    student.args = IterableSimpleNamespace(
        **model_args
    )

    # =================================================
    # CUDA
    # =================================================

    student = student.cuda()

    # =================================================
    # Train mode
    # =================================================

    student.train()

    # =================================================
    # Final verification
    # =================================================

    student_head = find_segmentation_head(
        student
    )

    final_head_nc = getattr(
        student_head,
        "nc",
        None
    )

    print()
    print("================")
    print("YOLO Student ready")
    print("================")

    print(
        "Model:",
        type(student)
    )

    print(
        "Head:",
        type(student.model[-1])
    )

    print(
        "Head nc:",
        student.model[-1].nc
    )

    print(
        "names:",
        student.names
    )

    print(
        "args.nc:",
        student.args.nc
    )

    print(
        "parameters:",
        sum(
            p.numel()
            for p in student.parameters()
        )
    )

    # =================================================
    # Hard verification
    # =================================================

    if final_head_nc != NUM_CLASSES:

        raise RuntimeError(
            "FINAL CHECK FAILED:\n"
            f"Student head nc = {final_head_nc}, "
            f"expected {NUM_CLASSES}"
        )

    if len(student.names) != NUM_CLASSES:

        raise RuntimeError(
            "FINAL CHECK FAILED:\n"
            f"Number of class names = "
            f"{len(student.names)}, "
            f"expected {NUM_CLASSES}"
        )

    if student.args.nc != NUM_CLASSES:

        raise RuntimeError(
            "FINAL CHECK FAILED:\n"
            f"student.args.nc = {student.args.nc}, "
            f"expected {NUM_CLASSES}"
        )

    print()
    print(
        "================================================"
    )
    print(
        "FINAL 4-CLASS STUDENT CHECK SUCCESS"
    )
    print(
        "================================================"
    )

    return student


# =====================================================
# create trainer
# =====================================================

def create_trainer():

    # =================================================
    # YOLO Student
    # =================================================

    student = build_four_class_student()

    # =================================================
    # YOLO feature hook
    # =================================================

    print()
    print("================================================")
    print("REGISTER YOLO FEATURE HOOK")
    print("================================================")

    hook = YOLOFeatureHook(
        student,
        layers=[
            15,
            18,
            21
        ]
    )

    hook.register()

    print(
        "YOLO hook ready"
    )

    # =================================================
    # SAM3 Teacher
    # =================================================

    print()
    print("================================================")
    print("LOAD SAM3 TEACHER")
    print("================================================")

    if not os.path.exists(SAM3_PATH):

        raise FileNotFoundError(
            "SAM3 model not found:\n"
            f"{SAM3_PATH}"
        )

    if not os.path.exists(BPE_PATH):

        raise FileNotFoundError(
            "SAM3 BPE tokenizer not found:\n"
            f"{BPE_PATH}"
        )

    teacher = SAM3Teacher(
        SAM3_PATH,
        BPE_PATH,
        device=DEVICE,
        img_size=1008,
        fp16=True
    )

    print(
        "SAM3 ready"
    )

    # =================================================
    # Feature adapters
    # =================================================

    print()
    print("================================================")
    print("CREATE FEATURE ADAPTERS")
    print("================================================")

    adapters = torch.nn.ModuleList(
        [
            FeatureAdapter(
                64,
                256
            ),

            FeatureAdapter(
                128,
                256
            ),

            FeatureAdapter(
                256,
                256
            )
        ]
    )

    adapters = adapters.cuda()

    print(
        "Feature adapters ready"
    )

    # =================================================
    # Feature loss
    # =================================================

    feature_loss = FeatureLoss()

    print(
        "Feature loss ready"
    )

    # =================================================
    # Optimizer
    # =================================================

    params = []

    params += list(
        student.parameters()
    )

    params += list(
        adapters.parameters()
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=1e-4,
        weight_decay=5e-4
    )

    print()
    print("Optimizer ready")

    # =================================================
    # Distillation trainer
    # =================================================

    trainer = DistillTrainer(
        teacher,
        student,
        adapters,
        feature_loss,
        optimizer,
        hook,
        lambda_feature=1.0,
        device=DEVICE,
        debug=False,
    )

    # =================================================
    # Final trainer verification
    # =================================================

    final_student_head = find_segmentation_head(
        trainer.student
    )

    print()
    print("================================================")
    print("DISTILLATION TRAINER READY")
    print("================================================")

    print()
    print("Final Student:")

    print(
        "   model type =",
        type(trainer.student)
    )

    print(
        "   head type =",
        type(final_student_head)
    )

    print(
        "   head nc =",
        final_student_head.nc
    )

    print(
        "   names =",
        trainer.student.names
    )

    print(
        "   args.nc =",
        trainer.student.args.nc
    )

    # =================================================
    # Hard verification
    # =================================================

    if final_student_head.nc != NUM_CLASSES:

        raise RuntimeError(
            "Trainer Student head nc is incorrect:\n"
            f"actual = {final_student_head.nc}\n"
            f"expected = {NUM_CLASSES}"
        )

    if len(trainer.student.names) != NUM_CLASSES:

        raise RuntimeError(
            "Trainer Student class names are incorrect:\n"
            f"actual = {len(trainer.student.names)}\n"
            f"expected = {NUM_CLASSES}"
        )

    if trainer.student.args.nc != NUM_CLASSES:

        raise RuntimeError(
            "Trainer Student args.nc is incorrect:\n"
            f"actual = {trainer.student.args.nc}\n"
            f"expected = {NUM_CLASSES}"
        )

    print()
    print(
        "================================================"
    )
    print(
        "4-CLASS DISTILLATION TRAINER CHECK SUCCESS"
    )
    print(
        "================================================"
    )

    return trainer


