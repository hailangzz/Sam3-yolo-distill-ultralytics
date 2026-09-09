# -*- coding: utf-8 -*-

from ultralytics.models.sam import SAM3SemanticPredictor


overrides = {

    "conf": 0.25,

    "task": "segment",

    "mode": "predict",

    "model": "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/sam3.pt",

}


predictor = SAM3SemanticPredictor(
    overrides=overrides
)
predictor.bpe_path = "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/bpe_simple_vocab_16e6.txt.gz"

predictor.setup_model()


model = predictor.model


print("======================")
print("SAM3 MODEL")
print("======================")


print(model)



print("======================")
print("SAM3 MODULE TREE")
print("======================")


for name, module in model.named_modules():

    print(name, ":", module.__class__.__name__)