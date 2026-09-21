from ultralytics.models.sam import SAM3SemanticPredictor

MODEL_PATH = "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/sam3.pt"
BPE_PATH = "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/bpe_simple_vocab_16e6.txt.gz"

overrides = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "model": MODEL_PATH,
    "save": True,
}

predictor = SAM3SemanticPredictor(overrides=overrides)

# 手动设置 SAM3 tokenizer 的 BPE 文件
predictor.bpe_path = BPE_PATH

print("[INFO] Model:", MODEL_PATH)
print("[INFO] BPE:", BPE_PATH)

predictor.set_image(
    "/data/database/aws_origin_sample/images/"
    "UT-A10XCNA00014A007/20260915/exist/"
    "1789439120669_UT-A10XCNA00014A007_carpet_detect_exist_0.742.jpg"
)

results = predictor(text=["person", "bus", "glasses"])

print(results)