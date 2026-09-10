
# -*- coding: utf-8 -*-

"""
SAM3 Feature Distillation Trainer

Teacher:
SAM3 text conditioned

Student:
YOLOv8-seg

Training:

    image
      |
      +----------------------+
      |                      |
      v                      v
    SAM3 Teacher          YOLO Student
      |                      |
      |                      +----> YOLO Prediction
      |                      |
      |                      +----> Hook Features
      |                               |
      |                               v
      |                            Adapter
      |                               |
      +-------------------------------+
                      |
                      v
                 Feature Loss

    YOLO Loss
         |
         v

    Total Loss =
        YOLO Loss
        + lambda_feature * Feature Loss


IMPORTANT:
This version intentionally uses pure FP32 training
for YOLO Student + Adapter.

SAM3 Teacher may still internally use FP16,
depending on the implementation of SAM3Teacher.

No autocast.
No GradScaler.
No AMP.

Purpose:
1. Verify whether the previous Inf gradient problem
   came from AMP / FP16 backward.
2. Establish a numerically stable FP32 baseline.
"""

import torch
import torch.nn.functional as F


class DistillTrainer:

    def __init__(
        self,
        teacher,
        student,
        adapters,
        feature_loss,
        optimizer,
        yolo_hook,
        lambda_feature=1.0,  # weight for feature loss,特征通道蒸馏，所采用的特征学习率系数。系数越大，特征蒸馏的权重越大，学生网络会更注重学习教师网络的特征表示，从而提高特征对齐的效果。系数越小，特征蒸馏的权重越小，学生网络会更注重自身的损失函数，从而提高任务性能。通常需要根据具体任务和数据集进行调节。
        device="cuda",
        debug=False,
    ):

        self.device = device

        self.teacher = teacher

        self.student = student

        self.adapters = adapters

        self.feature_loss = feature_loss

        self.optimizer = optimizer

        self.yolo_hook = yolo_hook

        self.lambda_feature = lambda_feature

        self.debug = debug

        # =================================================
        # IMPORTANT
        # =================================================
        # This version intentionally DOES NOT use:
        #
        #   autocast()
        #   GradScaler()
        #
        # YOLO Student and Adapter are trained using FP32.
        # =================================================

        # =================================================
        # freeze SAM3
        # =================================================

        self.teacher.eval()

        for p in self.teacher.parameters():

            p.requires_grad = False

        # =================================================
        # student train
        # =================================================

        self.student.train()

        for p in self.student.parameters():

            p.requires_grad = True

        # =================================================
        # adapter train
        # =================================================

        for adapter in self.adapters:

            adapter.train()

            for p in adapter.parameters():

                p.requires_grad = True

        # =================================================
        # DEBUG information
        # =================================================

        if self.debug:

            self.print_trainable_parameters()

    # ==================================================
    # DEBUG: trainable parameters
    # ==================================================

    def print_trainable_parameters(self):

        print("")
        print("=" * 80)
        print("TRAINABLE PARAMETER DEBUG")
        print("=" * 80)

        student_total = 0
        student_trainable = 0

        for name, p in self.student.named_parameters():

            num = p.numel()

            student_total += num

            if p.requires_grad:

                student_trainable += num

        print(
            "Student total parameters:",
            student_total
        )

        print(
            "Student trainable parameters:",
            student_trainable
        )

        adapter_total = 0
        adapter_trainable = 0

        for adapter in self.adapters:

            for p in adapter.parameters():

                num = p.numel()

                adapter_total += num

                if p.requires_grad:

                    adapter_trainable += num

        print(
            "Adapter total parameters:",
            adapter_total
        )

        print(
            "Adapter trainable parameters:",
            adapter_trainable
        )

        print(
            "lambda_feature:",
            self.lambda_feature
        )

        print(
            "training precision: FP32"
        )

        print("=" * 80)

    # ==================================================
    # DEBUG: tensor statistics
    # ==================================================

    @staticmethod
    def tensor_statistics(name, tensor):

        if tensor is None:

            print(
                f"[DEBUG] {name}: None"
            )

            return

        x = tensor.detach().float()

        print(
            f"[DEBUG] {name}: "
            f"shape={tuple(x.shape)} "
            f"dtype={tensor.dtype} "
            f"device={tensor.device} "
            f"mean={x.mean().item():.6f} "
            f"std={x.std().item():.6f} "
            f"min={x.min().item():.6f} "
            f"max={x.max().item():.6f} "
            f"abs_mean={x.abs().mean().item():.6f} "
            f"finite={torch.isfinite(x).all().item()}"
        )

    # ==================================================
    # DEBUG: feature list
    # ==================================================

    def debug_feature_list(self, name, features):

        print("")

        print(
            f"[DEBUG] {name}"
        )

        for i, f in enumerate(features):

            self.tensor_statistics(
                f"{name}[{i}]",
                f
            )

    # ==================================================
    # build prompt batch
    # ==================================================

    def build_prompt_batch(self, batch):

        """
        Convert instance prompts to image prompts.

        input:

            prompts:
            [
                "carpet",
                "wire",
                "liquid"
            ]

            batch_idx:
            tensor([
                0,
                0,
                1
            ])

        output:

            [
                [
                    "carpet",
                    "wire"
                ],

                [
                    "liquid"
                ]
            ]
        """

        prompts = batch["prompts"]

        batch_idx = batch["batch_idx"]

        batch_size = batch["img"].shape[0]

        image_prompts = []

        for img_id in range(batch_size):

            current_prompts = []

            ids = torch.where(
                batch_idx == img_id
            )[0]

            for idx in ids:

                idx = idx.item()

                current_prompts.append(
                    prompts[idx]
                )

            # ==========================================
            # 去重
            # ==========================================

            current_prompts = list(
                dict.fromkeys(current_prompts)
            )

            # ==========================================
            # 空目标
            # ==========================================

            if len(current_prompts) == 0:

                current_prompts.append(
                    "background"
                )

            image_prompts.append(
                current_prompts
            )

        return image_prompts

    # ==================================================
    # SAM3 forward
    # ==================================================

    def forward_teacher(
        self,
        images,
        prompt_batch,
    ):

        batch_outputs = []

        B = images.shape[0]

        for i in range(B):

            img = images[i:i + 1]

            prompt_features = []

            # ==========================================
            # 每个 prompt 单独运行 SAM3
            # ==========================================

            for prompt in prompt_batch[i]:

                feature = self.teacher(
                    img,
                    [prompt]
                )

                prompt_features.append(
                    feature
                )

            if len(prompt_features) == 0:

                raise RuntimeError(
                    f"No prompt features for image {i}"
                )

            # ==========================================
            # 多 prompt feature fusion
            #
            # torch.max()
            # ==========================================

            fused = []

            for level in range(
                len(prompt_features[0])
            ):

                level_features = []

                for f in prompt_features:

                    level_features.append(
                        f[level]
                    )

                merged = torch.cat(
                    level_features,
                    dim=0
                )

                level_feature = torch.max(
                    merged,
                    dim=0,
                    keepdim=True
                )[0]

                fused.append(
                    level_feature
                )

            batch_outputs.append(
                fused
            )

        # ==========================================
        # batch concat
        # ==========================================

        outputs = []

        for level in range(
            len(batch_outputs[0])
        ):

            outputs.append(
                torch.cat(
                    [
                        x[level]
                        for x in batch_outputs
                    ],
                    dim=0
                )
            )

        return outputs

    # ==================================================
    # gradient flatten
    # ==================================================

    @staticmethod
    def flatten_aligned_gradients(
        gradients,
        parameters,
    ):

        vectors = []

        for grad, param in zip(
            gradients,
            parameters
        ):

            if grad is None:

                vectors.append(
                    torch.zeros(
                        param.numel(),
                        device=param.device,
                        dtype=torch.float32,
                    )
                )

            else:

                vectors.append(
                    grad.detach()
                    .float()
                    .reshape(-1)
                )

        if len(vectors) == 0:

            return None

        return torch.cat(
            vectors,
            dim=0
        )

    # ==================================================
    # backward gradient norm
    # ==================================================

    @staticmethod
    def gradient_norm(gradients):

        total = 0.0

        for g in gradients:

            if g is not None:

                total += (
                    g.detach()
                    .float()
                    .pow(2)
                    .sum()
                    .item()
                )

        return total ** 0.5

    # ==================================================
    # module gradient norm
    # ==================================================

    @staticmethod
    def module_gradient_norm(module):

        total = 0.0

        for p in module.parameters():

            if p.grad is not None:

                g = p.grad.detach()

                finite_g = torch.where(
                    torch.isfinite(g),
                    g,
                    torch.zeros_like(g)
                )

                total += (
                    finite_g.float()
                    .pow(2)
                    .sum()
                    .item()
                )

        return total ** 0.5

    # ==================================================
    # check gradients
    # ==================================================

    def check_gradients(self):

        total = 0.0

        nan_count = 0

        inf_count = 0

        parameter_count = 0

        nan_parameters = []

        inf_parameters = []

        for name, p in self.student.named_parameters():

            if p.grad is None:

                continue

            parameter_count += 1

            g = p.grad.detach()

            has_nan = torch.isnan(g).any().item()

            has_inf = torch.isinf(g).any().item()

            if has_nan:

                nan_count += 1

                nan_parameters.append(
                    name
                )

            if has_inf:

                inf_count += 1

                inf_parameters.append(
                    name
                )

            # ==========================================
            # 只使用 finite gradient 计算 norm
            # ==========================================

            finite_g = torch.where(
                torch.isfinite(g),
                g,
                torch.zeros_like(g)
            )

            total += (
                finite_g.float()
                .pow(2)
                .sum()
                .item()
            )

        total_norm = total ** 0.5

        return {
            "norm": total_norm,

            "nan_count": nan_count,

            "inf_count": inf_count,

            "parameter_count": parameter_count,

            "nan_parameters": nan_parameters,

            "inf_parameters": inf_parameters,
        }

    # ==================================================
    # gradient coverage
    # ==================================================

    @staticmethod
    def gradient_coverage(
        gradients,
        parameters,
    ):

        total_parameters = len(parameters)

        used_parameters = sum(
            g is not None
            for g in gradients
        )

        total_numel = sum(
            p.numel()
            for p in parameters
        )

        used_numel = sum(
            p.numel()
            for p, g in zip(
                parameters,
                gradients
            )
            if g is not None
        )

        if total_parameters > 0:

            parameter_ratio = (
                used_parameters
                /
                total_parameters
            )

        else:

            parameter_ratio = 0.0

        if total_numel > 0:

            numel_ratio = (
                used_numel
                /
                total_numel
            )

        else:

            numel_ratio = 0.0

        return {
            "total_parameters":
                total_parameters,

            "used_parameters":
                used_parameters,

            "parameter_ratio":
                parameter_ratio,

            "total_numel":
                total_numel,

            "used_numel":
                used_numel,

            "numel_ratio":
                numel_ratio,
        }

    # ==================================================
    # train step
    # ==================================================

    def train_step(self, batch):

        # =================================================
        # 0. optimizer
        # =================================================

        self.optimizer.zero_grad(
            set_to_none=True
        )

        images = batch["img"]

        # =================================================
        # DEBUG
        # =================================================

        debug = self.debug

        if debug:

            print("")
            print("=" * 80)
            print("TRAIN STEP DEBUG")
            print("=" * 80)

            self.tensor_statistics(
                "images",
                images
            )

        # =================================================
        # 1. prompt
        # =================================================

        prompt_batch = self.build_prompt_batch(
            batch
        )

        if debug:

            print("")
            print(
                "[DEBUG] prompt_batch:"
            )

            for i, prompts in enumerate(
                prompt_batch
            ):

                print(
                    f"  image {i}:",
                    prompts
                )

        # =================================================
        # 2. SAM3 teacher
        # =================================================

        with torch.no_grad():

            teacher_features = (
                self.forward_teacher(
                    images,
                    prompt_batch
                )
            )

        teacher_features = [
            x.detach()
            for x in teacher_features
        ]

        # =================================================
        # select SAM3 levels
        # =================================================

        if len(teacher_features) < 4:

            raise RuntimeError(
                "SAM3 teacher feature number error: "
                f"expected at least 4, "
                f"actual={len(teacher_features)}"
            )

        sam3_features = [
            teacher_features[1],
            teacher_features[2],
            teacher_features[3],
        ]

        # =================================================
        # DEBUG teacher
        # =================================================

        if debug:

            self.debug_feature_list(
                "SAM3 features",
                sam3_features
            )

        # =================================================
        # 3. YOLO forward
        #
        # IMPORTANT:
        #
        # NO autocast
        #
        # Pure FP32
        # =================================================

        self.yolo_hook.clear()

        preds = self.student(
            images
        )

        # =================================================
        # student features
        # =================================================

        student_features = (
            self.yolo_hook.get_features()
        )

        if len(student_features) != 3:

            raise RuntimeError(
                "YOLO feature number error: "
                "expected=3 "
                f"actual={len(student_features)}"
            )

        # =================================================
        # DEBUG student
        # =================================================

        if debug:

            self.debug_feature_list(
                "YOLO features",
                student_features
            )

            for i, f in enumerate(
                student_features
            ):

                if not f.requires_grad:

                    raise RuntimeError(
                        f"YOLO feature {i} "
                        "does not require grad"
                    )

        # =================================================
        # 4. Adapter
        #
        # Pure FP32
        # =================================================

        adapted = []

        for i, f in enumerate(
            student_features
        ):

            out = self.adapters[i](
                f,
                target_size=sam3_features[i].shape[-2:]
            )

            adapted.append(
                out
            )

        # =================================================
        # Feature loss 使用 FP32
        # =================================================

        adapted = [
            x.float()
            for x in adapted
        ]

        sam3_features = [
            x.float()
            for x in sam3_features
        ]

        # =================================================
        # DEBUG adapter
        # =================================================

        if debug:

            self.debug_feature_list(
                "Adapted features",
                adapted
            )

        # =================================================
        # 5. Feature loss
        # =================================================

        loss_feature = self.feature_loss(
            adapted,
            sam3_features
        )

        # =================================================
        # feature loss check
        # =================================================

        if not torch.isfinite(
            loss_feature
        ):

            raise RuntimeError(
                "Feature loss is NaN or Inf: "
                f"{loss_feature.item()}"
            )

        # =================================================
        # 6. YOLO loss
        #
        # Pure FP32
        # =================================================

        yolo_result = self.student.loss(
            batch,
            preds=preds
        )

        if isinstance(
            yolo_result,
            tuple
        ):

            loss_yolo = yolo_result[0]

        else:

            loss_yolo = yolo_result

        if loss_yolo.ndim > 0:

            loss_yolo = loss_yolo.sum()

        # =================================================
        # YOLO loss check
        # =================================================

        if not torch.isfinite(
            loss_yolo
        ):

            raise RuntimeError(
                "YOLO loss is NaN or Inf: "
                f"{loss_yolo.item()}"
            )

        # =================================================
        # 7. total loss
        # =================================================

        loss = (
            loss_yolo
            +
            self.lambda_feature
            *
            loss_feature
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Total loss is NaN or Inf"
            )

        # =================================================
        # DEBUG loss
        # =================================================

        if debug:

            print("")
            print(
                "[DEBUG] LOSS"
            )

            print(
                "  loss_yolo    =",
                f"{loss_yolo.item():.8f}"
            )

            print(
                "  loss_feature =",
                f"{loss_feature.item():.8f}"
            )

            print(
                "  lambda       =",
                self.lambda_feature
            )

            print(
                "  feature contribution =",
                f"{self.lambda_feature * loss_feature.item():.8f}"
            )

            print(
                "  total_loss   =",
                f"{loss.item():.8f}"
            )

        # =================================================
        # 8. Gradient analysis
        # =================================================

        student_params = [
            p
            for p in self.student.parameters()
            if p.requires_grad
        ]

        # =================================================
        # YOLO gradient
        # =================================================

        grads_yolo = torch.autograd.grad(
            loss_yolo,
            student_params,
            retain_graph=True,
            allow_unused=True,
        )

        # =================================================
        # Feature gradient
        # =================================================

        grads_feature = torch.autograd.grad(
            loss_feature,
            student_params,
            retain_graph=True,
            allow_unused=True,
        )

        # =================================================
        # gradient norm
        # =================================================

        grad_yolo_norm = (
            self.gradient_norm(
                grads_yolo
            )
        )

        grad_feature_norm = (
            self.gradient_norm(
                grads_feature
            )
        )

        # =================================================
        # lambda adjusted feature gradient
        # =================================================

        grad_feature_weighted = (
            self.lambda_feature
            *
            grad_feature_norm
        )

        # =================================================
        # gradient ratio
        # =================================================

        if grad_yolo_norm > 1e-12:

            grad_ratio = (
                grad_feature_weighted
                /
                grad_yolo_norm
            )

        else:

            grad_ratio = float(
                "inf"
            )

        # =================================================
        # aligned gradient vectors
        # =================================================

        g_yolo = (
            self.flatten_aligned_gradients(
                grads_yolo,
                student_params
            )
        )

        g_feature = (
            self.flatten_aligned_gradients(
                grads_feature,
                student_params
            )
        )

        # =================================================
        # gradient cosine
        # =================================================

        if (
            g_yolo is not None
            and g_feature is not None
            and g_yolo.numel() > 0
            and g_feature.numel() > 0
            and g_yolo.numel()
            == g_feature.numel()
            and grad_yolo_norm > 1e-12
            and grad_feature_norm > 1e-12
        ):

            grad_cosine = F.cosine_similarity(
                g_yolo.unsqueeze(0),
                g_feature.unsqueeze(0),
                dim=1,
            ).item()

        else:

            grad_cosine = 0.0

        # =================================================
        # gradient coverage
        # =================================================

        yolo_coverage = (
            self.gradient_coverage(
                grads_yolo,
                student_params
            )
        )

        feature_coverage = (
            self.gradient_coverage(
                grads_feature,
                student_params
            )
        )

        # =================================================
        # DEBUG gradient
        # =================================================

        if debug:

            print("")
            print(
                "[DEBUG] GRADIENT ANALYSIS"
            )

            print(
                "  student parameter count =",
                len(student_params)
            )

            print(
                "  grad_yolo             =",
                f"{grad_yolo_norm:.6e}"
            )

            print(
                "  grad_feature          =",
                f"{grad_feature_norm:.6e}"
            )

            print(
                "  lambda * grad_feature =",
                f"{grad_feature_weighted:.6e}"
            )

            print(
                "  feature/yolo ratio    =",
                f"{grad_ratio:.6f}"
            )

            print(
                "  gradient cosine       =",
                f"{grad_cosine:.6f}"
            )

            print("")
            print(
                "  YOLO gradient coverage:"
            )

            print(
                "    parameters =",
                f"{yolo_coverage['used_parameters']}/"
                f"{yolo_coverage['total_parameters']}"
            )

            print(
                "    parameter ratio =",
                f"{yolo_coverage['parameter_ratio']:.4f}"
            )

            print(
                "    numel =",
                f"{yolo_coverage['used_numel']}/"
                f"{yolo_coverage['total_numel']}"
            )

            print(
                "    numel ratio =",
                f"{yolo_coverage['numel_ratio']:.4f}"
            )

            print("")
            print(
                "  Feature gradient coverage:"
            )

            print(
                "    parameters =",
                f"{feature_coverage['used_parameters']}/"
                f"{feature_coverage['total_parameters']}"
            )

            print(
                "    parameter ratio =",
                f"{feature_coverage['parameter_ratio']:.4f}"
            )

            print(
                "    numel =",
                f"{feature_coverage['used_numel']}/"
                f"{feature_coverage['total_numel']}"
            )

            print(
                "    numel ratio =",
                f"{feature_coverage['numel_ratio']:.4f}"
            )

            print("")
            print(
                "  aligned vector length =",
                g_yolo.numel()
                if g_yolo is not None
                else 0
            )

        # =================================================
        # 9. Backward
        #
        # Pure FP32
        # =================================================

        loss.backward()

        # =================================================
        # 10. Gradient check
        # =================================================

        gradient_info = (
            self.check_gradients()
        )

        if debug:

            print("")
            print(
                "[DEBUG] TOTAL GRADIENT"
            )

            print(
                "  norm =",
                f"{gradient_info['norm']:.6e}"
            )

            print(
                "  parameter_count =",
                gradient_info["parameter_count"]
            )

            print(
                "  nan_count =",
                gradient_info["nan_count"]
            )

            print(
                "  inf_count =",
                gradient_info["inf_count"]
            )

            # ==========================================
            # NaN parameter names
            # ==========================================

            if gradient_info["nan_count"] > 0:

                print("")
                print(
                    "  NaN parameters:"
                )

                for name in gradient_info[
                    "nan_parameters"
                ]:

                    print(
                        "   ",
                        name
                    )

            # ==========================================
            # Inf parameter names
            # ==========================================

            if gradient_info["inf_count"] > 0:

                print("")
                print(
                    "  Inf parameters:"
                )

                for name in gradient_info[
                    "inf_parameters"
                ]:

                    print(
                        "   ",
                        name
                    )

        # =================================================
        # 11. gradient safety check
        # =================================================

        if gradient_info["nan_count"] > 0:

            raise RuntimeError(
                "NaN gradient detected"
            )

        if gradient_info["inf_count"] > 0:

            raise RuntimeError(
                "Inf gradient detected"
            )

        # =================================================
        # 12. parameter update debug
        # =================================================

        debug_param = None

        debug_param_name = None

        debug_param_before = None

        if debug:

            for name, p in self.student.named_parameters():

                if p.requires_grad:

                    debug_param = p

                    debug_param_name = name

                    debug_param_before = (
                        p.detach()
                        .clone()
                    )

                    break

        # =================================================
        # 13. optimizer step
        #
        # Pure FP32
        # =================================================

        self.optimizer.step()

        # =================================================
        # 14. parameter update
        # =================================================

        param_update = 0.0

        if (
            debug
            and debug_param is not None
        ):

            param_update = (
                (
                    debug_param.detach()
                    -
                    debug_param_before
                )
                .abs()
                .mean()
                .item()
            )

            print("")
            print(
                "[DEBUG] PARAMETER UPDATE"
            )

            print(
                "  parameter:",
                debug_param_name
            )

            print(
                "  mean update:",
                f"{param_update:.6e}"
            )

            if param_update == 0:

                print(
                    "  WARNING: parameter "
                    "did not change!"
                )

        # =================================================
        # 15. return
        # =================================================

        return {
            "loss":
                loss.detach().item(),

            "loss_yolo":
                loss_yolo.detach().item(),

            "loss_feature":
                loss_feature.detach().item(),

            "grad_yolo":
                grad_yolo_norm,

            "grad_feature":
                grad_feature_norm,

            "grad_feature_weighted":
                grad_feature_weighted,

            "grad_ratio":
                grad_ratio,

            "grad_cosine":
                grad_cosine,

            "total_grad":
                gradient_info["norm"],

            "param_update":
                param_update,

            "yolo_grad_parameter_ratio":
                yolo_coverage[
                    "parameter_ratio"
                ],

            "feature_grad_parameter_ratio":
                feature_coverage[
                    "parameter_ratio"
                ],

            "yolo_grad_numel_ratio":
                yolo_coverage[
                    "numel_ratio"
                ],

            "feature_grad_numel_ratio":
                feature_coverage[
                    "numel_ratio"
                ],
        }

