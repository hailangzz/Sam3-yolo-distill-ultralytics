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


# 指定 tokenizer 文件
predictor.bpe_path = (
    "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/"
    "bpe_simple_vocab_16e6.txt.gz"
)


# 初始化模型
predictor.setup_model()


print("================ predictor")
print(type(predictor))


print("================ model")
print(type(predictor.model))


print("================ model keys")

print(
    predictor.model.__dict__.keys()
)

print("================ sub modules")

print(
    predictor.model._modules.keys()
)

print("================ backbone")

print(
    predictor.model.backbone
)