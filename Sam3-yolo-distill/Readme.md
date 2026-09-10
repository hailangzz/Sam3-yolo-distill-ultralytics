现在基于 **Ultralytics 8.3.237 + SAM3 Teacher + YOLOv8-seg Student Feature Distillation** 设计的工程结构如下：

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
SAM3 → YOLOv8-Seg Feature Distillation
蒸馏训练使用说明
============================================================

项目：
    Sam3-yolo-distill

作用：
    使用 SAM3 作为 Teacher，
    YOLOv8-Seg 作为 Student，
    通过中间特征蒸馏，使 YOLOv8-Seg 学习 SAM3 的特征表示。

------------------------------------------------------------
一、基本训练结构
------------------------------------------------------------

                     ┌──────────────────┐
                     │      输入图像      │
                     └────────┬─────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌───────────────┐           ┌───────────────┐
        │   SAM3 Teacher │           │ YOLOv8 Student│
        │    冻结参数     │           │    可训练参数   │
        └───────┬───────┘           └───────┬───────┘
                │                           │
                ▼                           ▼
          SAM3 多尺度特征              YOLOv8 多尺度特征
                │                           │
                │                    ┌──────┴──────┐
                │                    │   Adapter   │
                │                    │ 特征尺寸/通道 │
                │                    │     对齐      │
                │                    └──────┬──────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                       Feature Loss
                              │
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
             YOLO Loss               Feature Loss
                 │                         │
                 └────────────┬────────────┘
                              ▼
                         Total Loss

Total Loss：

    total_loss = loss_yolo + lambda_feature * loss_feature


============================================================
二、项目目录
============================================================

当前项目：

    /data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill

主要目录：

    configs/
        classes.py
        distill.py

    datasets/
        carpet.yaml
        yolo_dataset.py

    teacher/
        sam3_teacher.py

    hooks/
        sam3_feature_hook.py
        yolo_feature_hook.py
        yolo_hook.py

    modules/
        adapters.py

    losses/
        feature_loss.py

    distill_trainer.py

    train_distill.py
    train.py

    models/
        sam3.pt
        bpe_simple_vocab_16e6.txt.gz

    tests/
        test_real_train.py
        test_distill_trainer.py
        test_feature_loss.py
        test_sam3_teacher.py
        ...


============================================================
三、环境要求
============================================================

当前验证环境：

    Python:
        3.10.9

    PyTorch:
        2.6.0+cu124

    CUDA:
        12.4

    Ultralytics:
        8.3.237

    GPU:
        NVIDIA RTX 4060 Ti 8GB

训练前建议检查：

    python -c "import torch; print(torch.__version__)"
    python -c "import torch; print(torch.cuda.is_available())"

应该看到：

    torch 2.6.0+cu124
    True


============================================================
四、类别 Prompt 配置
============================================================

类别定义位于：

    configs/classes.py

当前配置：

    CLASS_PROMPTS = {
        0: "carpet",
        1: "wire",
        2: "liquid",
        3: "plastic bag"
    }

这里的 class_id 必须和 YOLO 数据集中的类别 ID 对应。

例如：

    0 → carpet
    1 → wire
    2 → liquid
    3 → plastic bag

如果增加或修改类别，需要同步修改：

    1. configs/classes.py
    2. 数据集 YAML
    3. YOLO 标签
    4. Student 模型类别数量
    5. 训练/验证代码


============================================================
五、数据集要求
============================================================

目前使用 YOLO Segmentation 数据格式。

典型结构：

    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    │
    └── labels/
        ├── train/
        └── val/

标签格式：

    class_id x1 y1 x2 y2 x3 y3 ...


例如：

    0 0.10 0.20 0.20 0.20 0.25 0.30 ...


数据集 YAML 示例：

    path: /data/your_dataset

    train: images/train
    val: images/val

    names:
        0: carpet
        1: wire
        2: liquid
        3: plastic bag


============================================================
六、Teacher：SAM3
============================================================

SAM3 模型：

    models/sam3.pt

Tokenizer：

    models/bpe_simple_vocab_16e6.txt.gz

SAM3 Teacher 为冻结状态。

训练过程中：

    Teacher 不进行参数更新。

Teacher 的作用：

    输入：

        image
        text prompt

    输出：

        多尺度视觉特征


当前 Teacher 输出大致为：

    Level 0:
        [B, 256, 288, 288]

    Level 1:
        [B, 256, 144, 144]

    Level 2:
        [B, 256, 72, 72]

    Level 3:
        [B, 256, 36, 36]

当前蒸馏主要使用：

    Level 1
    Level 2
    Level 3


============================================================
七、Student：YOLOv8-Seg
============================================================

Student 是 YOLOv8-Seg。

当前 Hook 捕获的主要特征：

    Feature 1:
        [B, 64, 80, 80]

    Feature 2:
        [B, 128, 40, 40]

    Feature 3:
        [B, 256, 20, 20]


Teacher 和 Student 的：

    通道数
    特征图尺寸

并不一致。

因此需要 Adapter 进行特征对齐。


============================================================
八、Adapter
============================================================

Adapter 的主要作用：

    YOLO Feature
          ↓
    Channel Alignment
          ↓
    Spatial Alignment
          ↓
    Teacher Feature Size


最终 Student Feature 被转换为：

    [B, 256, 144, 144]
    [B, 256, 72, 72]
    [B, 256, 36, 36]

然后和 SAM3 Teacher Feature 计算 Feature Loss。


============================================================
九、Prompt 机制
============================================================

每张图片根据其 YOLO 标注中的类别自动生成 Prompt。

例如：

图片 A：

    carpet
    wire

则：

    prompt_batch[A] = [
        "carpet",
        "wire"
    ]

图片 B：

    只有 carpet

则：

    prompt_batch[B] = [
        "carpet"
    ]

没有目标的图片：

    prompt_batch[B] = [
        "background"
    ]


同一张图片如果存在多个类别：

    Teacher 会分别计算：

        SAM3(image, "carpet")
        SAM3(image, "wire")

然后进行多 Prompt 特征融合。


============================================================
十、Feature Loss
============================================================

当前训练同时计算两个 Loss：

    1. YOLO Supervised Loss
    2. Feature Distillation Loss


最终：

    total_loss =
        loss_yolo
        + lambda_feature * loss_feature


其中：

    lambda_feature

控制蒸馏 Loss 对 Student 的影响。


============================================================
十一、lambda_feature 如何设置
============================================================

DistillTrainer 中：

    class DistillTrainer:

        def __init__(
            self,
            teacher,
            student,
            adapters,
            feature_loss,
            optimizer,
            yolo_hook,
            lambda_feature=1.0,
            device="cuda",
            debug=False,
        ):


默认：

    lambda_feature = 1.0


训练时推荐从外部传入：

    trainer = DistillTrainer(
        teacher=teacher,
        student=student,
        adapters=adapters,
        feature_loss=feature_loss,
        optimizer=optimizer,
        yolo_hook=yolo_hook,
        lambda_feature=5.0,
        device="cuda",
        debug=True,
    )


这样可以方便进行不同蒸馏强度实验。


============================================================
十二、推荐的 lambda 实验
============================================================

建议至少进行：

    lambda = 0
    lambda = 1
    lambda = 5
    lambda = 10


含义：

    lambda = 0

        不进行特征蒸馏。

        相当于 YOLOv8-Seg baseline。


    lambda = 1

        当前基础蒸馏方案。


    lambda = 5

        中等强度蒸馏。


    lambda = 10

        强蒸馏。


实验时不要只看：

    total loss

应该重点比较：

    Precision
    Recall
    mAP50
    mAP50-95
    Segmentation mAP
    各类别 AP
    Hard Case 性能
    推理速度


============================================================
十三、当前训练的一个重要现象
============================================================

目前观察到：

    YOLO gradient

明显大于：

    Feature gradient


例如某次训练：

    grad_yolo:
        177.54

    grad_feature:
        0.1271

比例：

    feature / yolo ≈ 0.000849


也就是说：

    lambda_feature = 1

时，

    Feature Loss 对 Student 参数的直接梯度影响非常小。


因此：

    lambda = 5

或者：

    lambda = 10

值得进一步实验。


注意：

不能仅根据 Loss 数值大小判断蒸馏强弱。

更重要的是：

    Feature Gradient
    YOLO Gradient
    Gradient Ratio
    Gradient Cosine


============================================================
十四、AMP 混合精度
============================================================

当前训练支持 CUDA AMP。

Student forward：

    autocast("cuda", dtype=torch.float16)

Feature Loss：

    使用 FP32

这样可以降低显存和计算成本，同时保持 Feature Loss 的数值稳定性。


正确的 AMP 反向流程：

    self.scaler.scale(loss).backward()

    self.scaler.unscale_(self.optimizer)

    # 此处进行 gradient check

    self.scaler.step(self.optimizer)

    self.scaler.update()


不要使用：

    loss.backward()

然后再：

    scaler.unscale_(optimizer)

否则会出现：

    AssertionError:
    Attempted unscale_ but _scale is None


============================================================
十五、训练命令
============================================================

进入项目：

    cd /data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill


使用当前 Python：

    /home/chenkejing/anaconda3/bin/python


运行测试训练：

    /home/chenkejing/anaconda3/bin/python \
        tests/test_real_train.py


============================================================
十六、正式训练前建议
============================================================

第一步：

    检查 CUDA

    python -c "import torch; print(torch.cuda.is_available())"


第二步：

    检查 SAM3

    python tests/test_sam3_teacher.py


第三步：

    检查 Dataset

    python tests/test_yolo_dataset.py


第四步：

    检查 Feature Loss

    python tests/test_feature_loss.py


第五步：

    检查完整 Trainer

    python tests/test_distill_trainer.py


第六步：

    再运行：

    python tests/test_real_train.py


不要一开始就直接跑 100～300 epochs。


============================================================
十七、训练过程中重点观察
============================================================

正常情况下应该观察：

    loss_yolo
        ↓
    持续下降


    loss_feature
        ↓
    前期下降
        ↓
    后期逐渐稳定


如果出现：

    loss = NaN

或者：

    loss_feature = NaN

或者：

    gradient = Inf

需要立即停止并检查：

    AMP
    Feature Adapter
    Feature Loss
    Learning Rate


当前训练应该检查：

    images finite
    teacher features finite
    student features finite
    adapted features finite
    losses finite
    gradients finite


============================================================
十八、Gradient Debug
============================================================

debug=True 时，可以观察：

    grad_yolo
    grad_feature
    feature/yolo ratio
    gradient cosine
    gradient coverage


例如：

    grad_yolo = 177.54

    grad_feature = 0.1271

    ratio = 0.000849


解释：

    Feature Loss 当前对 Student 的梯度贡献
    相对于 YOLO Loss 非常小。


如果：

    ratio ≈ 0

说明：

    蒸馏信号非常弱。


如果：

    ratio 非常大

例如：

    > 1

则需要警惕：

    Feature Loss 压制 YOLO Loss。


============================================================
十九、训练参数建议
============================================================

初始实验建议：

    imgsz:
        640

    batch:
        根据 GPU 显存设置

    epochs:
        100

    AMP:
        True

    lambda_feature:
        1 / 5 / 10 分别实验


对于 RTX 4060 Ti 8GB：

    batch size 不宜盲目增大。

如果显存不足：

    优先降低 batch size

而不是关闭 AMP。


============================================================
二十、蒸馏训练的正确实验流程
============================================================

建议按照：

    ┌──────────────────────┐
    │  YOLOv8-Seg Baseline │
    │      lambda = 0      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │     Distillation     │
    │      lambda = 1      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │     Distillation     │
    │      lambda = 5      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │     Distillation     │
    │      lambda = 10     │
    └──────────┬───────────┘
               │
               ▼
          Val Evaluation
               │
               ▼
       选择最佳模型


不要根据：

    training loss

单独决定最终模型。


最终应该根据验证集和实际机器人场景综合决定。


============================================================
二十一、最终模型能否继续普通 YOLO 微调
============================================================

当前这种方案的核心原则是：

    Student 仍然是 YOLOv8-Seg

蒸馏过程中：

    Teacher
        ↓
    Feature Loss
        ↓
    Student

不会因为增加：

    Teacher
    Feature Loss
    Adapter

而改变最终 Student 的 YOLOv8-Seg 推理结构。


因此训练完成后：

    Student checkpoint

仍然应该可以作为普通 YOLOv8-Seg 模型继续：

    Fine-tuning
    Validation
    Inference
    Deployment


注意：

最终部署时只需要：

    YOLOv8-Seg Student


不需要：

    SAM3
    Teacher
    Adapter
    Feature Loss
    DistillTrainer


因此：

    SAM3 蒸馏属于训练阶段的辅助机制，

而：

    YOLOv8-Seg Student

才是最终部署模型。


============================================================
二十二、推荐的模型生命周期
============================================================

原始模型：

    YOLOv8-Seg
          │
          ▼
    SAM3 Distillation
          │
          ▼
    Distilled YOLOv8-Seg
          │
          ├──── 普通 Fine-tuning
          │
          ├──── Hard Case Fine-tuning
          │
          ├──── 数据增强训练
          │
          └──── 最终部署


因此后续可以继续：

    distilled_model.pt

→ 普通 YOLO 训练

→ Hard Case 微调

→ 最终机器人部署。


============================================================
二十三、常见问题
============================================================

问题 1：

    为什么 Feature Loss 很小？

回答：

    Loss 数值小并不代表蒸馏一定很强。

    应同时观察：

        Feature Gradient
        YOLO Gradient
        Gradient Ratio


问题 2：

    为什么 Feature Loss 下降以后趋于稳定？

回答：

    可能意味着：

        Adapter 已经完成主要特征空间对齐，

    或：

        Student backbone 已经难以进一步匹配 Teacher。


问题 3：

    为什么 Feature Gradient 只覆盖部分参数？

回答：

    Feature Loss 来自被 Hook 的 backbone feature。

    因此它不会直接经过所有 YOLO 参数。

    这是当前结构下正常现象。


问题 4：

    为什么 total loss 下降？

回答：

    当前：

        YOLO Loss
        +
        Feature Loss

    同时优化。

    只要 YOLO Loss 和 Feature Loss 都正常下降，
    一般说明训练流程工作正常。


问题 5：

    最终部署是否需要 SAM3？

回答：

    不需要。

    SAM3 只参与训练阶段。

    最终机器人端只部署 YOLOv8-Seg Student。


============================================================
二十四、推荐的实验记录
============================================================

每次实验建议记录：

    Experiment:
        distill_lambda5_v1

    Model:
        YOLOv8-Seg

    Teacher:
        SAM3

    Dataset:
        carpet / wire / liquid / plasticbag

    Image Size:
        640

    Batch:
        8

    Epoch:
        100

    lambda_feature:
        5

    Best Epoch:
        xxx

    Precision:
        xxx

    Recall:
        xxx

    mAP50:
        xxx

    mAP50-95:
        xxx

    Segmentation mAP:
        xxx

    Inference Speed:
        xxx FPS

    GPU:
        RTX 4060 Ti 8GB


============================================================
二十五、推荐的最终训练流程
============================================================

完整流程：

    ① 准备 YOLO Segmentation 数据集

            ↓

    ② 配置 classes.py

            ↓

    ③ 准备 SAM3

            ↓

    ④ 初始化 YOLOv8-Seg Student

            ↓

    ⑤ 初始化 SAM3 Teacher

            ↓

    ⑥ 注册 YOLO Feature Hook

            ↓

    ⑦ 初始化 Adapter

            ↓

    ⑧ 初始化 Feature Loss

            ↓

    ⑨ 初始化 DistillTrainer

            ↓

    ⑩ 设置 lambda_feature

            ↓

    ⑪ AMP 蒸馏训练

            ↓

    ⑫ 保存 Student checkpoint

            ↓

    ⑬ 使用独立 Val 集评估

            ↓

    ⑭ 与原始 YOLOv8-Seg 对比

            ↓

    ⑮ 选择最佳模型

            ↓

    ⑯ 普通 YOLO / Hard Case Fine-tuning

            ↓

    ⑰ 最终机器人部署


============================================================
二十六、最重要的注意事项
============================================================

1. Teacher 必须保持冻结。

2. Student 才是最终需要保存和部署的模型。

3. Adapter 只服务于训练阶段。

4. lambda_feature 不应该只看 Loss 数值决定。

5. 必须比较 lambda=0 的 YOLO baseline。

6. 最终模型必须在独立验证集上评估。

7. 不要只看 mAP50。

8. 对机器人项目尤其需要关注：

       False Positive
       False Negative
       Hard Case
       小目标
       遮挡
       复杂背景
       实际现场数据


9. 蒸馏训练结束后，应检查：

       Student 是否可以独立加载
       Student 是否可以独立 inference
       Student 是否可以普通 YOLO fine-tuning
       Student 是否可以正常导出/部署


============================================================
二十七、快速开始
============================================================

最简单的使用方式：

    cd /data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill

    /home/chenkejing/anaconda3/bin/python \
        tests/test_real_train.py


修改蒸馏强度：

    lambda_feature=1.0

或者：

    lambda_feature=5.0

或者：

    lambda_feature=10.0


训练结束后：

    1. 保存 Student checkpoint
    2. 在 val 数据集上评估
    3. 与 YOLO baseline 对比
    4. 检查 Hard Case
    5. 决定是否进入后续 Fine-tuning


============================================================
结束
============================================================
"""


def print_usage():
    print(__doc__)


if __name__ == "__main__":
    print_usage()



### 注意 pycharm Edit设置 ###
Working directory ： /data/Sam3-yolo-distill-ultralytics/ultralytics

Environment variables： PYTHONPATH=/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill:/data/Sam3-yolo-distill-ultralytics