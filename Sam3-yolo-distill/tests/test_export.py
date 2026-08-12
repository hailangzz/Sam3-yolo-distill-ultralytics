# -*- coding:utf-8 -*-


from utils.export_yolo import export_student



CHECKPOINT = (

"/data/ultralytics/"
"Sam3-yolo-distill/"
"weights/"
"epoch_20.pt"

)



MODEL_YAML = (

"/data/ultralytics/"
"Sam3-yolo-distill/"
"yolov8n-seg.yaml"

)



OUTPUT = (

"/data/ultralytics/"
"Sam3-yolo-distill/"
"weights/"
"best_distill.pt"

)



export_student(

    CHECKPOINT,

    MODEL_YAML,

    OUTPUT

)



print(
"EXPORT TEST PASSED"
)