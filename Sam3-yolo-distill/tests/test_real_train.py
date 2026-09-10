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

IMAGE_DIR = "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_carpet_database/images/train"


LABEL_DIR = "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_carpet_database/labels/train"


DEVICE = "cuda"


EPOCHS = 100


BATCH_SIZE = 8


CHECKPOINT_DIR = "./checkpoints"


# =====================================================
# DEBUG CONFIG
# =====================================================

# 是否开启 debug
DEBUG = True


# 只在第一个 epoch 的第一个 batch
# 打印完整 debug 信息
DEBUG_FIRST_BATCH_ONLY = True


# 每多少个 batch 打印一次普通训练信息
PRINT_EVERY = 10


# =====================================================
# create checkpoint directory
# =====================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================
# checkpoint save
# =====================================================


def save_checkpoint(
    trainer,
    epoch,
    loss,
    filename,
):

    save_path = os.path.join(CHECKPOINT_DIR, filename)

    # =================================================
    # Student
    # =================================================

    student_state = trainer.student.state_dict()

    # =================================================
    # Adapter
    # =================================================

    adapter_state = []

    for adapter in trainer.adapters:

        adapter_state.append(adapter.state_dict())

    # =================================================
    # optimizer
    # =================================================

    optimizer_state = trainer.optimizer.state_dict()

    # =================================================
    # scaler
    # =================================================

    scaler_state = None

    if hasattr(trainer, "scaler"):

        scaler_state = trainer.scaler.state_dict()

    # =================================================
    # checkpoint
    # =================================================

    checkpoint = {
        # ---------------------------------------------
        # training state
        # ---------------------------------------------
        "epoch": epoch,
        "loss": loss,
        # ---------------------------------------------
        # student model
        # ---------------------------------------------
        "student": student_state,
        # ---------------------------------------------
        # adapters
        # ---------------------------------------------
        "adapters": adapter_state,
        # ---------------------------------------------
        # optimizer
        # ---------------------------------------------
        "optimizer": optimizer_state,
        # ---------------------------------------------
        # AMP scaler
        # ---------------------------------------------
        "scaler": scaler_state,
        # ---------------------------------------------
        # distillation config
        # ---------------------------------------------
        "lambda_feature": trainer.lambda_feature,
    }

    # =================================================
    # save
    # =================================================

    torch.save(checkpoint, save_path)

    print("checkpoint saved:", save_path)


# =====================================================
# create trainer
# =====================================================

print("==============================")
print("create trainer")


trainer = create_trainer()


# =====================================================
# DEBUG mode
# =====================================================

trainer.debug = False


print("trainer ready")


# =====================================================
# print model information
# =====================================================

print("")
print("==============================")
print("MODEL INFORMATION")


student_parameter_count = sum(p.numel() for p in trainer.student.parameters())


student_trainable_count = sum(
    p.numel() for p in trainer.student.parameters() if p.requires_grad
)


adapter_parameter_count = sum(
    p.numel() for adapter in trainer.adapters for p in adapter.parameters()
)


adapter_trainable_count = sum(
    p.numel()
    for adapter in trainer.adapters
    for p in adapter.parameters()
    if p.requires_grad
)


print("student parameters:", student_parameter_count)


print("student trainable:", student_trainable_count)


print("adapter parameters:", adapter_parameter_count)


print("adapter trainable:", adapter_trainable_count)


print("lambda_feature:", trainer.lambda_feature)


print("==============================")


# =====================================================
# dataset
# =====================================================

print("")
print("==============================")
print("create dataset")


dataset = YOLODataset(image_dir=IMAGE_DIR, label_dir=LABEL_DIR, img_size=640)


print("dataset size:", len(dataset))


assert len(dataset) > 0


# =====================================================
# dataloader
# =====================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=dataset.collate_fn,
)


print("dataloader ready")


print("batch number:", len(loader))


# =====================================================
# training
# =====================================================

print("")
print("==============================")
print("START TRAINING")
print("==============================")


best_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    print("")
    print("=" * 80)

    print(f"Epoch {epoch}/{EPOCHS}")

    print("=" * 80)

    # =================================================
    # epoch statistics
    # =================================================

    total_loss = 0.0

    total_yolo = 0.0

    total_feature = 0.0

    total_grad_yolo = 0.0

    total_grad_feature = 0.0

    total_grad_ratio = 0.0

    total_grad_cosine = 0.0

    total_param_update = 0.0

    batch_count = 0

    # =================================================
    # dataloader
    # =================================================

    for batch_idx, batch in enumerate(loader):

        # =================================================
        # CUDA
        # =================================================

        for k, v in batch.items():

            if torch.is_tensor(v):

                batch[k] = v.to(DEVICE, non_blocking=True)

        # =================================================
        # DEBUG
        # =================================================

        if DEBUG:

            if DEBUG_FIRST_BATCH_ONLY:

                trainer.debug = epoch == 1 and batch_idx == 0

            else:

                trainer.debug = True

        else:

            trainer.debug = False

        # =================================================
        # train step
        # =================================================

        result = trainer.train_step(batch)

        # =================================================
        # losses
        # =================================================

        loss = result["loss"]

        loss_yolo = result["loss_yolo"]

        loss_feature = result["loss_feature"]

        # =================================================
        # accumulate
        # =================================================

        total_loss += float(loss)

        total_yolo += float(loss_yolo)

        total_feature += float(loss_feature)

        # =================================================
        # gradient debug
        # =================================================

        if "grad_yolo" in result:

            total_grad_yolo += float(result["grad_yolo"])

        if "grad_feature" in result:

            total_grad_feature += float(result["grad_feature"])

        if "grad_ratio" in result:

            total_grad_ratio += float(result["grad_ratio"])

        if "grad_cosine" in result:

            total_grad_cosine += float(result["grad_cosine"])

        if "param_update" in result:

            total_param_update += float(result["param_update"])

        batch_count += 1

        # =================================================
        # batch print
        # =================================================

        if batch_idx % PRINT_EVERY == 0 or (epoch == 1 and batch_idx == 0):

            print("")

            print(
                f"[Epoch {epoch}] "
                f"batch "
                f"{batch_idx}/{len(loader)} "
                f"loss="
                f"{loss:.4f} "
                f"yolo="
                f"{loss_yolo:.4f} "
                f"feature="
                f"{loss_feature:.4f}"
            )

            # -----------------------------------------
            # debug metrics
            # -----------------------------------------

            if "grad_yolo" in result:

                print(f"    grad_yolo=" f"{result['grad_yolo']:.4e}")

            if "grad_feature" in result:

                print(f"    grad_feature=" f"{result['grad_feature']:.4e}")

            if "grad_ratio" in result:

                print(f"    feature/yolo=" f"{result['grad_ratio']:.6f}")

            if "grad_cosine" in result:

                print(f"    grad_cosine=" f"{result['grad_cosine']:.6f}")

            if "param_update" in result:

                print(f"    param_update=" f"{result['param_update']:.4e}")

    # =====================================================
    # epoch statistics
    # =====================================================

    if batch_count == 0:

        raise RuntimeError("No batches were processed.")

    avg_loss = total_loss / batch_count

    avg_yolo = total_yolo / batch_count

    avg_feature = total_feature / batch_count

    avg_grad_yolo = total_grad_yolo / batch_count

    avg_grad_feature = total_grad_feature / batch_count

    avg_grad_ratio = total_grad_ratio / batch_count

    avg_grad_cosine = total_grad_cosine / batch_count

    avg_param_update = total_param_update / batch_count

    # =====================================================
    # print epoch result
    # =====================================================

    print("")
    print("=" * 80)

    print(f"Epoch {epoch} finished")

    print("-" * 80)

    print(f"loss           : {avg_loss:.6f}")

    print(f"loss_yolo      : {avg_yolo:.6f}")

    print(f"loss_feature   : {avg_feature:.6f}")

    print("-" * 80)

    print(f"grad_yolo      : " f"{avg_grad_yolo:.6e}")

    print(f"grad_feature   : " f"{avg_grad_feature:.6e}")

    print(f"feature/yolo   : " f"{avg_grad_ratio:.6f}")

    print(f"grad_cosine    : " f"{avg_grad_cosine:.6f}")

    print(f"param_update   : " f"{avg_param_update:.6e}")

    print("=" * 80)

    # =====================================================
    # save epoch checkpoint
    # =====================================================

    save_checkpoint(trainer, epoch, avg_loss, f"epoch_{epoch}.pt")

    # =====================================================
    # latest checkpoint
    # =====================================================

    save_checkpoint(trainer, epoch, avg_loss, "latest.pt")

    # =====================================================
    # best checkpoint
    # =====================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        save_checkpoint(trainer, epoch, avg_loss, "best.pt")

        print("best checkpoint updated.")


# =====================================================
# training finished
# =====================================================

print("")
print("=" * 80)

print("REAL TRAIN FINISHED")

print("best loss:", best_loss)

print("=" * 80)
