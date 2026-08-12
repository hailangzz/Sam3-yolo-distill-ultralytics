import torch

from ultralytics.models.sam import SAM3SemanticPredictor


# ======================
# config
# ======================

overrides = {

    "conf": 0.25,

    "task": "segment",

    "mode": "predict",

    "model":
    "/data/ultralytics/Sam3-yolo-distill/models/sam3.pt",

}



# ======================
# predictor
# ======================

predictor = SAM3SemanticPredictor(
    overrides=overrides
)


predictor.bpe_path = (
    "/data/ultralytics/Sam3-yolo-distill/models/"
    "bpe_simple_vocab_16e6.txt.gz"
)


predictor.setup_model()



model = predictor.model.cuda()

model.eval()



print("================ model")

print(type(model))



# ======================
# vision backbone
# ======================

vision = model.backbone.vision_backbone


vision.cuda()

vision.eval()


print("================ vision")

print(type(vision))



# ======================
# image size
# ======================

imgsz = [1008,1008]


vision.set_imgsz(
    imgsz
)



# ======================
# input
# ======================

image = torch.randn(
    1,
    3,
    1008,
    1008,
    device="cuda",
    dtype=torch.float16
)



vision.half()



# ======================
# forward
# ======================

with torch.inference_mode():

    feat = vision(
        image
    )



print("================ output")

print(type(feat))



sam3_features, sam3_pos, sam2_features, sam2_pos = feat



print("\n========= SAM3 feature =========")


for i,x in enumerate(sam3_features):

    print(
        i,
        x.shape,
        x.dtype
    )



print("\n========= SAM3 pos =========")


for i,x in enumerate(sam3_pos):

    print(
        i,
        x.shape
    )



print("\n========= SAM2 feature =========")


if sam2_features is None:

    print("None")

else:

    for i,x in enumerate(sam2_features):

        print(
            i,
            x.shape
        )


print("\nDone")