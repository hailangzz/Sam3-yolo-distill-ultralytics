from teacher.sam3_teacher import SAM3Teacher
import torch


teacher=SAM3Teacher(
    "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/sam3.pt",
    "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/models/bpe_simple_vocab_16e6.txt.gz"
)


img=torch.randn(
    1,
    3,
    1008,
    1008
).cuda()


out=teacher(
    img,
    ["carpet"]
)


for i,x in enumerate(out):

    print(i,x.shape)