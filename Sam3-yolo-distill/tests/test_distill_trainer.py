import torch


from teacher.sam3_teacher import SAM3Teacher

from modules.adapters import FeatureAdapter

from losses.feature_loss import FeatureLoss



sam3_path = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/models/sam3.pt"
)


bpe_path = (
    "/data/ultralytics/"
    "Sam3-yolo-distill/models/"
    "bpe_simple_vocab_16e6.txt.gz"
)



teacher = SAM3Teacher(
    sam3_path,
    bpe_path
)



# =========================
# adapter
# =========================

adapters = [

    FeatureAdapter(
        256,
        128
    ),

    FeatureAdapter(
        256,
        256
    ),

    FeatureAdapter(
        256,
        512
    )

]

adapters = [
    x.cuda().half()
    for x in adapters
]

loss_fn = FeatureLoss()



print("================")
print("components ready")



image=torch.randn(
    1,
    3,
    1008,
    1008,
    device="cuda"
)



with torch.no_grad():

    teacher_feat = teacher(
        image
    )


teacher_feat = [
    x.detach().clone()
    for x in teacher_feat
]


print("================")
print("teacher")


for i,x in enumerate(teacher_feat):

    print(
        i,
        x.shape
    )



print("================")
print("adapter output")


student_like=[]


for i in range(3):

    y = adapters[i](
        teacher_feat[i+1]
    )

    student_like.append(y)


    print(
        i,
        y.shape
    )

# PYTHONPATH=/data/ultralytics:/data/ultralytics/Sam3-yolo-distill  python tests/test_distill_trainer.py



