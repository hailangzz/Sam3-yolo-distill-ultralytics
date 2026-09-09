# -*- coding: utf-8 -*-


"""
Test mAP evaluator
"""


from ultralytics import YOLO


from validation.map_evaluator import MAPEvaluator



MODEL_PATH = (

"/data/Sam3-yolo-distill-ultralytics/"
"Sam3-yolo-distill/"
"yolov8n-seg.pt"

)



DATA_YAML = (

"/data/Sam3-yolo-distill-ultralytics/"
"Sam3-yolo-distill/"
"tests/data.yaml"

)



DEVICE="cuda"



print("================")
print("load model")


model = YOLO(
    MODEL_PATH
)



print(
"model loaded"
)



evaluator = MAPEvaluator(

    model,

    DATA_YAML,

    device=DEVICE

)



print("================")
print("run evaluation")


metrics = evaluator.evaluate()



assert isinstance(
    metrics,
    dict
)


assert len(metrics)>0



print("================")
print(
"MAP EVALUATOR TEST PASSED"
)