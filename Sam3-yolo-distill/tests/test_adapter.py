import torch


from modules.adapters import (
    MultiScaleFeatureAdapter
)



adapter = MultiScaleFeatureAdapter(

    teacher_channels=[
        256,
        256,
        256,
        256
    ],

    student_channels=[
        128,
        128,
        256,
        512
    ]

)



features=[

torch.randn(
1,256,288,288
),

torch.randn(
1,256,144,144
),

torch.randn(
1,256,72,72
),

torch.randn(
1,256,36,36
)

]



out = adapter(features)



for i,x in enumerate(out):

    print(
        i,
        x.shape
    )


# PYTHONPATH=/data/ultralytics:/data/ultralytics/Sam3-yolo-distill python tests/test_adapter.py