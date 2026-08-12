# -*- coding:utf-8 -*-


import torch

from torch.utils.data import DataLoader


from train_distill import create_trainer

from datasets.yolo_dataset import YOLODataset

from validation.evaluator import DistillEvaluator



IMAGE_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/images/train"
)


LABEL_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/labels/train"
)



print("================")
print("create trainer")


trainer = create_trainer()



print("================")
print("create evaluator")



evaluator = DistillEvaluator(

    trainer.teacher,

    trainer.student,

    trainer.adapters,

    trainer.feature_loss,

    trainer.yolo_hook

)



dataset = YOLODataset(

    IMAGE_DIR,

    LABEL_DIR,

    img_size=640

)



loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=False,

    num_workers=0,

    collate_fn=dataset.collate_fn

)



print("================")
print("run validation")



result = evaluator.evaluate(

    loader

)



print(result)



assert "val_loss" in result

assert result["val_loss"] > 0



print("================")

print(
"VALIDATION TEST PASSED"
)