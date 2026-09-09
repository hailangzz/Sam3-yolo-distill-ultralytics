from ultralytics.models.sam import SAM3SemanticPredictor


# overrides = {
#     "conf": 0.25,
#     "task": "segment",
#     "mode": "predict",
#     "model": "/home/chenkejing/Downloads/sam3.pt",
#     "save": True,
# }

overrides = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "model": "/home/chenkejing/Downloads/sam3.pt",
    "save": True,

    "project": "/data/Sam3-yolo-distill-ultralytics/runs/segment",
    "name": "sam3_text_segment",
    "exist_ok": True,
}

predictor = SAM3SemanticPredictor(
    overrides=overrides,
    bpe_path="/home/chenkejing/Downloads/bpe_simple_vocab_16e6.txt.gz"
)


image_path = "/home/chenkejing/Downloads/WireSegmentProject/spatial_location_val_images/wire_detect/UT-A10XCNA00815T006/20260811/null/1786437016385_UT-A10XCNA00815T006_wire_detect_null_0.000.jpg"


predictor.set_image(image_path)

# results = predictor(
#     text=["person", "bus", "glasses"]
# )


# predictor.set_image(image_path)
# results = predictor(
#     text=["person with red cloth", "person with blue cloth"]
# )


# predictor.set_image(image_path)
#
# results = predictor( text=["a person"] )

# results = predictor(
#     text=["wire on floor"]
# )

results = predictor( text=["floor"] )