import torch

from teacher.sam3_teacher import SAM3Teacher


teacher = SAM3Teacher(
    model_path="models/sam3.pt",
    bpe_path="models/bpe_simple_vocab_16e6.txt.gz",
    device="cuda"
)


image = torch.randn(
    1,
    3,
    640,
    640
).cuda()



features = teacher(

    image,

    [
        "carpet"
    ]

)



print(
    len(features)
)


for f in features:

    print(
        f.shape
    )