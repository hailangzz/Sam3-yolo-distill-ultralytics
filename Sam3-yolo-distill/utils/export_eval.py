# -*- coding: utf-8 -*-

"""
Build YOLO evaluator from current student model

Used during training validation
"""


from ultralytics import YOLO



def build_eval_model_from_student(
        student,
        model_yaml
):


    print("================")
    print("build eval model from student")


    model = YOLO(
        model_yaml
    )


    model.model.load_state_dict(
        student.state_dict(),
        strict=False
    )


    model.model.cuda()


    model.model.eval()


    print(
        "eval model ready"
    )


    return model