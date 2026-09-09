from ultralytics.models.sam import SAM3SemanticPredictor


overrides={

    "conf":0.25,
    "task":"segment",
    "mode":"predict",
    "model":"/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/sam3.pt",

}


predictor=SAM3SemanticPredictor(
    overrides=overrides
)


predictor.bpe_path="/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/bpe_simple_vocab_16e6.txt.gz"

predictor.setup_model()


model=predictor.model


print("================ transformer ================")

print(model.transformer)


print("================ children ================")


for name,module in model.transformer.named_children():

    print(
        name,
        type(module)
    )