import torch


from losses.feature_loss import FeatureLoss



criterion = FeatureLoss()



teacher = [

torch.randn(
1,128,288,288
),

torch.randn(
1,128,144,144
),

torch.randn(
1,256,72,72
),

torch.randn(
1,512,36,36
)

]



student=[

torch.randn(
1,128,288,288,
requires_grad=True
),

torch.randn(
1,128,144,144,
requires_grad=True
),

torch.randn(
1,256,72,72,
requires_grad=True
),

torch.randn(
1,512,36,36,
requires_grad=True
)

]



loss = criterion(
    student,
    teacher
)



print(
    "loss:",
    loss
)



loss.backward()



for x in student:

    print(
        x.grad is not None
    )

# PYTHONPATH=/data/ultralytics:/data/ultralytics/Sam3-yolo-distill python tests/test_feature_loss.py