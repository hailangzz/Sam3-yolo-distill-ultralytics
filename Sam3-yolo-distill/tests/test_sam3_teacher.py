import torch


from teacher.sam3_teacher import SAM3Teacher



teacher = SAM3Teacher(

    model_path=
    "/data/ultralytics/Sam3-yolo-distill/models/sam3.pt",


    bpe_path=
    "/data/ultralytics/Sam3-yolo-distill/models/bpe_simple_vocab_16e6.txt.gz"

)



teacher.cuda()



image = torch.randn(
    1,
    3,
    640,
    640
)


with torch.no_grad():

    feats = teacher(
        image
        .cuda()
    )



print("================")

print(type(feats))


for i,f in enumerate(feats):

    print(
        i,
        f.shape,
        f.dtype
    )


# cd /data/ultralytics/Sam3-yolo-distill
#
# PYTHONPATH=/data/ultralytics:/data/ultralytics/Sam3-yolo-distill python tests/test_sam3_teacher.py