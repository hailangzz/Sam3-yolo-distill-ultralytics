#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 批量自动标注

功能：
1. SAM3 模型只加载一次
2. 从 total_images_path_info.txt 逐张读取图片
3. 使用文本 Prompt 进行 SAM3 自动检测/分割
4. 将 SAM3 Mask 转换为 YOLO Segmentation Label
5. 有检测目标：
   - 生成 .txt Label
   - 将 Label 路径 + 检测到的 Prompt 写入 info 文件
6. 无检测目标：
   - 不生成 .txt Label
   - 不写入 info 文件
7. 已存在 Label 的图片可以跳过
8. 单张图片异常不会影响后续图片
9. 最后输出处理统计信息
10. 目标结构筛选：
    - rug or carpet
    - Cables or wires on the ground
    - Plastic sheets or plastic bags on the ground
    - Liquid stains on the ground

    对以上目标：
        y_center = (y_min + y_max) / 2

    如果：
        y_center < image_height / 2

    则过滤该目标，不作为有效 Mask。

    person 不参与该过滤。

11. 新增 CUDA 显存保护：
    - inference_mode
    - 每张图片结束后释放 Results
    - gc.collect()
    - torch.cuda.empty_cache()
    - CUDA OOM 自动清理并重试一次
    - 显存监控
    - expandable_segments
"""

# ============================================================
# 必须在 import torch 之前设置
# ============================================================

import os

# 减少长期运行过程中 CUDA 显存碎片问题
#
# 注意：
# 必须在 torch 初始化 CUDA allocator 之前设置。
#
# 如果你是直接：
#     python sam3_predict.py
#
# 那么这里设置最方便。
#
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)


# ============================================================
# Imports
# ============================================================

import gc
import traceback
from pathlib import Path

import torch
from PIL import Image

from ultralytics.models.sam import SAM3SemanticPredictor


# ============================================================
# CUDA 显存工具
# ============================================================

def cleanup_cuda_memory(
    reason="",
):
    """
    清理 Python 垃圾对象以及 PyTorch CUDA Cache。

    注意：
    torch.cuda.empty_cache() 不会删除仍然被 Python 引用的 Tensor。

    所以这里先：
        gc.collect()

    再：
        torch.cuda.empty_cache()
    """

    if reason:
        print(
            f"[GPU] 开始清理显存：{reason}"
        )

    # --------------------------------------------------------
    # Python GC
    # --------------------------------------------------------

    gc.collect()

    # --------------------------------------------------------
    # PyTorch CUDA Cache
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        # ----------------------------------------------------
        # synchronize
        #
        # 确保之前的 CUDA 操作完成。
        # ----------------------------------------------------

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    if reason:
        print_gpu_memory(
            prefix="清理完成"
        )


def print_gpu_memory(
    prefix="",
):
    """
    打印当前 GPU 显存状态。

    allocated：
        当前仍然被 PyTorch Tensor 使用的显存。

    reserved：
        PyTorch CUDA allocator 从 GPU 申请并保留的显存。

    reserved - allocated：
        当前 allocator 缓存但暂未使用的显存。

    max_allocated：
        当前进程历史最大 allocated。
    """

    if not torch.cuda.is_available():
        return

    try:

        allocated = (
            torch.cuda.memory_allocated()
            / 1024 ** 3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024 ** 3
        )

        max_allocated = (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
        )

        free, total = (
            torch.cuda.mem_get_info()
        )

        free = free / 1024 ** 3
        total = total / 1024 ** 3

        cached = (
            reserved - allocated
        )

        print(
            f"[GPU] {prefix} | "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB | "
            f"cached={cached:.2f} GB | "
            f"free={free:.2f} GB | "
            f"total={total:.2f} GB | "
            f"max_allocated={max_allocated:.2f} GB"
        )

    except Exception as e:

        print(
            f"[GPU] 获取显存信息失败：{e}"
        )


def reset_gpu_peak_memory():
    """
    重置 PyTorch 的历史峰值显存统计。
    """

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ============================================================
# 创建 SAM3 Predictor
# ============================================================

def create_predictor():
    """
    创建 SAM3 Semantic Predictor。

    注意：
    bpe_path 不能直接放进 overrides，
    否则 Ultralytics get_cfg() 会认为它是非法 YOLO 参数。
    """

    overrides = {
        "conf": CONF,
        "task": "segment",
        "mode": "predict",
        "model": MODEL_PATH,
        "save": False,
    }

    predictor = SAM3SemanticPredictor(
        overrides=overrides
    )

    # 必须单独设置
    predictor.bpe_path = BPE_PATH

    return predictor


# ============================================================
# 提前加载 SAM3 模型
# ============================================================

def warmup_predictor(
    predictor,
):
    """
    提前加载 SAM3 模型。

    SAM3SemanticPredictor 本身是 lazy loading，
    真正的模型通常在 set_image() 时才 setup_model()。

    这里主动调用 setup_model()，
    确保整个批处理过程中只加载一次模型。
    """

    print("=" * 80)
    print("[INFO] 正在加载 SAM3 模型...")
    print(f"[INFO] MODEL_PATH: {MODEL_PATH}")
    print(f"[INFO] BPE_PATH:   {BPE_PATH}")
    print("=" * 80)

    predictor.setup_model()

    print("[INFO] SAM3 模型加载完成")

    print_gpu_memory(
        prefix="模型加载完成"
    )

    print("=" * 80)


# ============================================================
# 读取图片路径
# ============================================================

def load_image_paths():
    """
    从 total_images_path_info.txt 中读取所有图片路径。

    每行一个图片路径。

    自动：
    - 去除空行
    - 去除首尾空格
    - 去除重复路径
    """

    if not TOTAL_IMAGES_INFO.exists():

        raise FileNotFoundError(
            f"图片路径信息文件不存在："
            f"{TOTAL_IMAGES_INFO}"
        )

    image_paths = []

    with TOTAL_IMAGES_INFO.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            image_path = line.strip()

            if not image_path:
                continue

            image_paths.append(
                Path(image_path)
            )

    # 去重，同时保持原始顺序
    image_paths = list(
        dict.fromkeys(image_paths)
    )

    return image_paths


# ============================================================
# 读取已经存在的 Label
# ============================================================

def load_existing_label_paths():
    """
    从 info 文件中读取已经记录过的 Label 路径。

    返回 set，用于快速判断。
    """

    existing_paths = set()

    if not INFO_OUTPUT_FILE.exists():
        return existing_paths

    with INFO_OUTPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            path = line.strip()

            if not path:
                continue

            # ------------------------------------------------
            # 兼容：
            #
            # /xxx/xxx.txt:person,rug or carpet
            #
            # 判断是否已经存在时，只取 ":" 前面的 Label 路径
            # ------------------------------------------------

            label_path = path.split(
                ":",
                1
            )[0].strip()

            existing_paths.add(
                label_path
            )

    return existing_paths


# ============================================================
# 获取 Label 路径
# ============================================================

def get_label_path(
    image_path,
):
    """
    根据图片路径生成 Label 路径。
    """

    LABEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        LABEL_OUTPUT_DIR
        / f"{image_path.stem}.txt"
    )


# ============================================================
# 获取图片尺寸
# ============================================================

def get_image_size(
    image_path,
    result=None,
):
    """
    获取图片原始尺寸。

    优先使用 SAM3 result.orig_shape。
    如果没有，则使用 PIL。
    """

    if result is not None:

        orig_shape = getattr(
            result,
            "orig_shape",
            None
        )

        if orig_shape is not None:

            height, width = (
                orig_shape[:2]
            )

            return (
                int(width),
                int(height)
            )

    with Image.open(
        image_path
    ) as image:

        width, height = image.size

    return width, height


# ============================================================
# 判断单个目标是否应该被过滤
# ============================================================

def should_filter_target(
    polygon,
    class_id,
    image_height,
):
    """
    判断当前 Mask 目标是否应该被过滤。
    """

    if class_id < 0:
        return False

    if class_id >= len(TEXT_PROMPTS):
        return False

    prompt = TEXT_PROMPTS[class_id]

    # --------------------------------------------------------
    # 不属于需要过滤的目标
    # --------------------------------------------------------

    if prompt not in FILTER_PROMPTS:
        return False

    # --------------------------------------------------------
    # Polygon 无效
    # --------------------------------------------------------

    if polygon is None:
        return False

    if len(polygon) < 3:
        return False

    # --------------------------------------------------------
    # 提取 y
    # --------------------------------------------------------

    y_values = []

    for point in polygon:

        if point is None:
            continue

        if len(point) < 2:
            continue

        y = float(point[1])

        y_values.append(y)

    if not y_values:
        return False

    # --------------------------------------------------------
    # y_min / y_max
    # --------------------------------------------------------

    y_min = min(y_values)
    y_max = max(y_values)

    # --------------------------------------------------------
    # Mask 中心
    # --------------------------------------------------------

    mask_y_center = (
        y_min + y_max
    ) / 2.0

    # --------------------------------------------------------
    # Image 中心
    # --------------------------------------------------------

    image_y_center = (
        image_height / 2.0
    )

    # --------------------------------------------------------
    # 判断
    # --------------------------------------------------------

    if mask_y_center < image_y_center:

        print(
            "[FILTER] 过滤目标："
            f"{prompt}"
        )

        print(
            f"[FILTER] y_min={y_min:.2f}, "
            f"y_max={y_max:.2f}, "
            f"mask_y_center={mask_y_center:.2f}, "
            f"image_y_center={image_y_center:.2f}"
        )

        return True

    return False


# ============================================================
# SAM3 Mask -> YOLO Segmentation
# ============================================================

def masks_to_yolo_labels(
    result,
    image_width,
    image_height,
):
    """
    将 SAM3 segmentation mask 转换成 YOLO Segmentation 格式。

    YOLO Segmentation：

        class_id x1 y1 x2 y2 x3 y3 ...

    坐标归一化到 0 ~ 1。
    """

    labels = []

    if result is None:
        return labels

    masks = getattr(
        result,
        "masks",
        None
    )

    boxes = getattr(
        result,
        "boxes",
        None
    )

    if masks is None:
        return labels

    if boxes is None:
        return labels

    polygons = getattr(
        masks,
        "xy",
        None
    )

    classes = getattr(
        boxes,
        "cls",
        None
    )

    if polygons is None:
        return labels

    if classes is None:
        return labels

    try:

        class_ids = (
            classes
            .cpu()
            .numpy()
            .astype(int)
        )

    except Exception:

        class_ids = (
            classes
            .numpy()
            .astype(int)
        )

    if len(polygons) == 0:
        return labels

    # --------------------------------------------------------
    # 遍历每一个 Mask
    # --------------------------------------------------------

    for polygon, class_id in zip(
        polygons,
        class_ids
    ):

        if polygon is None:
            continue

        if len(polygon) < 3:
            continue

        class_id = int(
            class_id
        )

        # ----------------------------------------------------
        # 结构过滤
        # ----------------------------------------------------

        if should_filter_target(
            polygon=polygon,
            class_id=class_id,
            image_height=image_height,
        ):

            continue

        # ----------------------------------------------------
        # 生成 YOLO Label
        # ----------------------------------------------------

        line = [
            str(class_id)
        ]

        for point in polygon:

            x = float(point[0])
            y = float(point[1])

            x_norm = (
                x / image_width
            )

            y_norm = (
                y / image_height
            )

            x_norm = max(
                0.0,
                min(1.0, x_norm)
            )

            y_norm = max(
                0.0,
                min(1.0, y_norm)
            )

            line.append(
                f"{x_norm:.6f}"
            )

            line.append(
                f"{y_norm:.6f}"
            )

        labels.append(
            " ".join(line)
        )

    return labels


# ============================================================
# 获取当前图片检测到的 Prompt
# ============================================================

def get_detected_prompts(
    result,
    image_height,
):
    """
    获取经过结构过滤之后的 Prompt。
    """

    detected_prompts = []

    if result is None:
        return detected_prompts

    boxes = getattr(
        result,
        "boxes",
        None
    )

    masks = getattr(
        result,
        "masks",
        None
    )

    if boxes is None:
        return detected_prompts

    if masks is None:
        return detected_prompts

    classes = getattr(
        boxes,
        "cls",
        None
    )

    polygons = getattr(
        masks,
        "xy",
        None
    )

    if classes is None:
        return detected_prompts

    if polygons is None:
        return detected_prompts

    try:

        class_ids = (
            classes
            .cpu()
            .numpy()
            .astype(int)
        )

    except Exception:

        class_ids = (
            classes
            .numpy()
            .astype(int)
        )

    for polygon, class_id in zip(
        polygons,
        class_ids
    ):

        class_id = int(
            class_id
        )

        # ----------------------------------------------------
        # 结构过滤
        # ----------------------------------------------------

        if should_filter_target(
            polygon=polygon,
            class_id=class_id,
            image_height=image_height,
        ):

            continue

        if class_id < 0:
            continue

        if class_id >= len(TEXT_PROMPTS):
            continue

        prompt = TEXT_PROMPTS[class_id]

        if prompt not in detected_prompts:

            detected_prompts.append(
                prompt
            )

    return detected_prompts


# ============================================================
# SAM3 单次推理
# ============================================================

def run_sam3_inference(
    predictor,
    image_path,
):
    """
    执行一次 SAM3 推理。

    使用 inference_mode，确保不会建立 autograd graph。
    """

    predictor.set_image(
        str(image_path)
    )

    # --------------------------------------------------------
    # 关键：
    # inference_mode 比 no_grad 更适合纯推理场景。
    # --------------------------------------------------------

    with torch.inference_mode():

        results = predictor(
            text=TEXT_PROMPTS
        )

    return results


# ============================================================
# 处理单张图片
# ============================================================

def process_one_image(
    predictor,
    image_path,
    existing_label_paths,
):
    """
    处理单张图片。

    返回：

        label_path, detected_prompts, status

    status：

        success
        skip
        no_target
        failed
    """

    results = None

    try:

        if not image_path.exists():

            print(
                f"[WARNING] 图片不存在："
                f"{image_path}"
            )

            return (
                None,
                [],
                "failed"
            )

        label_path = get_label_path(
            image_path
        )

        # ----------------------------------------------------
        # 已经存在 Label
        # ----------------------------------------------------

        if SKIP_EXISTING_LABEL:

            if label_path.exists():

                print(
                    f"[SKIP] Label 已存在："
                    f"{label_path}"
                )

                return (
                    label_path,
                    [],
                    "skip"
                )

            if str(label_path) in existing_label_paths:

                print(
                    f"[SKIP] Info 中已存在记录："
                    f"{label_path}"
                )

                return (
                    label_path,
                    [],
                    "skip"
                )

        # ----------------------------------------------------
        # SAM3 推理
        # ----------------------------------------------------

        print(
            f"\n[PROCESS] {image_path}"
        )

        results = run_sam3_inference(
            predictor=predictor,
            image_path=image_path,
        )

        # ----------------------------------------------------
        # 检查结果
        # ----------------------------------------------------

        if results is None:

            print(
                "[INFO] SAM3 没有返回结果"
            )

            return (
                None,
                [],
                "no_target"
            )

        # ----------------------------------------------------
        # 统一成 list
        # ----------------------------------------------------

        if not isinstance(
            results,
            (list, tuple)
        ):

            results = [results]

        all_labels = []

        all_detected_prompts = []

        # ----------------------------------------------------
        # 遍历 Result
        # ----------------------------------------------------

        for result in results:

            image_width, image_height = (
                get_image_size(
                    image_path,
                    result
                )
            )

            labels = masks_to_yolo_labels(
                result=result,
                image_width=image_width,
                image_height=image_height,
            )

            all_labels.extend(
                labels
            )

            detected_prompts = (
                get_detected_prompts(
                    result=result,
                    image_height=image_height,
                )
            )

            for prompt in detected_prompts:

                if prompt not in all_detected_prompts:

                    all_detected_prompts.append(
                        prompt
                    )

        # ----------------------------------------------------
        # 立即删除 SAM3 Results
        # ----------------------------------------------------

        # 注意：
        # 这里非常重要。
        #
        # 后面已经不再需要 results，
        # 所以不要让它继续存活到函数末尾。
        #

        del results
        results = None

        # ----------------------------------------------------
        # 没有有效目标
        # ----------------------------------------------------

        if not all_labels:

            print(
                f"[INFO] 没有有效目标："
                f"{image_path}"
            )

            print(
                "[INFO] SAM3 检测目标可能全部"
                "被结构筛选规则过滤"
            )

            return (
                None,
                [],
                "no_target"
            )

        # ----------------------------------------------------
        # 生成 Label
        # ----------------------------------------------------

        LABEL_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with label_path.open(
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(all_labels)
            )

            f.write("\n")

        print(
            f"[SUCCESS] Label 已生成："
            f"{label_path}"
        )

        print(
            f"[SUCCESS] 有效检测目标数量："
            f"{len(all_labels)}"
        )

        print(
            f"[SUCCESS] 有效 Prompt："
            f"{all_detected_prompts}"
        )

        return (
            label_path,
            all_detected_prompts,
            "success"
        )

    except torch.cuda.OutOfMemoryError:

        # ----------------------------------------------------
        # CUDA OOM
        # ----------------------------------------------------

        print(
            "\n[CUDA OOM] 当前图片推理发生显存不足："
            f"{image_path}"
        )

        print_gpu_memory(
            prefix="OOM 发生时"
        )

        # ----------------------------------------------------
        # 删除当前 Results
        # ----------------------------------------------------

        results = None

        # ----------------------------------------------------
        # 清理 CUDA Cache
        # ----------------------------------------------------

        cleanup_cuda_memory(
            reason="CUDA OOM"
        )

        # ----------------------------------------------------
        # 不在这里直接重试。
        #
        # 由 main() 决定是否重试，
        # 避免 process_one_image 内部无限重试。
        # ----------------------------------------------------

        return (
            None,
            [],
            "cuda_oom"
        )

    except Exception as e:

        print(
            f"\n[ERROR] 处理失败："
            f"{image_path}"
        )

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        traceback.print_exc()

        return (
            None,
            [],
            "failed"
        )

    finally:

        # ----------------------------------------------------
        # 无论成功、失败还是异常：
        #
        # 都主动清理 Python 临时对象。
        # ----------------------------------------------------

        results = None

        gc.collect()


# ============================================================
# CUDA OOM 重试
# ============================================================

def process_one_image_with_retry(
    predictor,
    image_path,
    existing_label_paths,
    max_retries=1,
):
    """
    CUDA OOM 专用恢复机制。

    第一次 OOM：
        cleanup_cuda_memory()
        ↓
        重试一次

    第二次仍然 OOM：
        判定 failed

    普通异常：
        不重试。
    """

    for attempt in range(
        max_retries + 1
    ):

        if attempt > 0:

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"[RETRY] CUDA OOM 后重新尝试："
                f"{image_path}"
            )

            print(
                f"[RETRY] 第 {attempt} 次重试"
            )

            print(
                "=" * 80
            )

            cleanup_cuda_memory(
                reason=f"第 {attempt} 次 OOM 重试前"
            )

        (
            label_path,
            detected_prompts,
            status
        ) = process_one_image(
            predictor=predictor,
            image_path=image_path,
            existing_label_paths=(
                existing_label_paths
            ),
        )

        # ----------------------------------------------------
        # 正常结束
        # ----------------------------------------------------

        if status != "cuda_oom":

            return (
                label_path,
                detected_prompts,
                status
            )

        # ----------------------------------------------------
        # OOM
        # ----------------------------------------------------

        print(
            f"[OOM] 第 {attempt + 1} 次推理 OOM"
        )

        cleanup_cuda_memory(
            reason="OOM 后清理"
        )

    # --------------------------------------------------------
    # 所有重试都失败
    # --------------------------------------------------------

    print(
        f"[ERROR] CUDA OOM 重试仍然失败："
        f"{image_path}"
    )

    return (
        None,
        [],
        "failed"
    )


# ============================================================
# 主函数
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("SAM3 批量自动标注")
    print("=" * 80)

    print(
        f"[INFO] 图片列表："
        f"{TOTAL_IMAGES_INFO}"
    )

    print(
        f"[INFO] Prompt："
        f"{TEXT_PROMPTS}"
    )

    print(
        f"[INFO] Label 目录："
        f"{LABEL_OUTPUT_DIR}"
    )

    print(
        f"[INFO] Info 文件："
        f"{INFO_OUTPUT_FILE}"
    )

    print(
        f"[INFO] SKIP_EXISTING_LABEL："
        f"{SKIP_EXISTING_LABEL}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 显存配置
    # --------------------------------------------------------

    print(
        "[INFO] CUDA allocator："
        f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 结构筛选规则
    # --------------------------------------------------------

    print(
        "[INFO] 目标结构筛选："
    )

    print(
        "[INFO] FILTER_PROMPTS："
        f"{list(FILTER_PROMPTS)}"
    )

    print(
        "[INFO] 筛选条件："
        "(y_min + y_max) / 2 < image_height / 2"
    )

    print(
        "[INFO] 满足条件的目标将被过滤"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 读取图片路径
    # --------------------------------------------------------

    image_paths = load_image_paths()

    total_count = len(
        image_paths
    )

    print(
        f"[INFO] 共读取 {total_count} 张图片"
    )

    if total_count == 0:

        print(
            "[WARNING] 没有需要处理的图片"
        )

        return

    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------

    LABEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 读取已有 Info
    # --------------------------------------------------------

    existing_label_paths = (
        load_existing_label_paths()
    )

    print(
        f"[INFO] Info 文件中已有 "
        f"{len(existing_label_paths)} 条记录"
    )

    # --------------------------------------------------------
    # 创建 SAM3 Predictor
    # --------------------------------------------------------

    predictor = create_predictor()

    # --------------------------------------------------------
    # 提前加载模型
    # --------------------------------------------------------

    warmup_predictor(
        predictor
    )

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    success_count = 0
    skip_count = 0
    no_target_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # 打开 Info 文件
    # --------------------------------------------------------

    with INFO_OUTPUT_FILE.open(
        "a",
        encoding="utf-8"
    ) as info_file:

        # ----------------------------------------------------
        # 顺序处理
        # ----------------------------------------------------

        for index, image_path in enumerate(
            image_paths,
            start=1
        ):

            print("\n")
            print("-" * 80)

            print(
                f"[{index}/{total_count}] "
                f"开始处理"
            )

            print(
                f"[IMAGE] {image_path}"
            )

            # ------------------------------------------------
            # 每隔 100 张打印一次显存
            # ------------------------------------------------

            if index % 100 == 0:

                print_gpu_memory(
                    prefix=f"处理 {index} 张之后"
                )

            # ------------------------------------------------
            # 单张图片处理
            # ------------------------------------------------

            (
                label_path,
                detected_prompts,
                status
            ) = process_one_image_with_retry(
                predictor=predictor,
                image_path=image_path,
                existing_label_paths=(
                    existing_label_paths
                ),
                max_retries=1,
            )

            # ------------------------------------------------
            # 成功
            # ------------------------------------------------

            if status == "success":

                success_count += 1

                prompt_info = ",".join(
                    detected_prompts
                )

                info_file.write(
                    f"{label_path}:{prompt_info}\n"
                )

                info_file.flush()

                existing_label_paths.add(
                    str(label_path)
                )

            # ------------------------------------------------
            # 已存在
            # ------------------------------------------------

            elif status == "skip":

                skip_count += 1

            # ------------------------------------------------
            # 无有效目标
            # ------------------------------------------------

            elif status == "no_target":

                no_target_count += 1

            # ------------------------------------------------
            # 失败
            # ------------------------------------------------

            elif status == "failed":

                failed_count += 1

            # ------------------------------------------------
            # 当前进度
            # ------------------------------------------------

            processed_count = (
                success_count
                + skip_count
                + no_target_count
                + failed_count
            )

            print(
                f"[PROGRESS] "
                f"{processed_count}/{total_count} | "
                f"success={success_count}, "
                f"skip={skip_count}, "
                f"no_target={no_target_count}, "
                f"failed={failed_count}"
            )

    # ========================================================
    # 最终统计
    # ========================================================

    print("\n")
    print("=" * 80)
    print("SAM3 批量自动标注完成")
    print("=" * 80)

    print(
        f"[RESULT] 总图片数："
        f"{total_count}"
    )

    print(
        f"[RESULT] 成功生成 Label："
        f"{success_count}"
    )

    print(
        f"[RESULT] 已存在跳过："
        f"{skip_count}"
    )

    print(
        f"[RESULT] 无有效目标："
        f"{no_target_count}"
    )

    print(
        f"[RESULT] 处理失败："
        f"{failed_count}"
    )

    print("-" * 80)

    print(
        f"[RESULT] Label 目录："
        f"{LABEL_OUTPUT_DIR}"
    )

    print(
        f"[RESULT] Info 文件："
        f"{INFO_OUTPUT_FILE}"
    )

    print("=" * 80)


# ============================================================
# 配置
# ============================================================

# SAM3 模型
MODEL_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/sam3.pt"
)

# SAM3 BPE tokenizer
BPE_PATH = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/models/bpe_simple_vocab_16e6.txt.gz"
)

# 图片路径列表
TOTAL_IMAGES_INFO = Path(
    "/data/database/aws_origin_sample/images/"
    "total_images_path_info.txt"
)

# ============================================================
# 文本 Prompt
# ============================================================

TEXT_PROMPTS = [
    "person",
    "rug or carpet",
    "Cables or wires on the ground",
    "Plastic sheets or plastic bags on the ground",
    "Liquid stains on the ground"
]

# 是否跳过已经存在的 Label
SKIP_EXISTING_LABEL = True

# SAM3 confidence
CONF = 0.25


# ============================================================
# 需要进行目标结构筛选的 Prompt
# ============================================================

FILTER_PROMPTS = {
    "rug or carpet",
    "Cables or wires on the ground",
    "Plastic sheets or plastic bags on the ground",
    "Liquid stains on the ground",
}


# ============================================================
# 输出路径
# ============================================================

LABEL_OUTPUT_DIR = (
    TOTAL_IMAGES_INFO.parent
    / "Sam3_auto_labels"
)

prompt_string = "_".join(
    TEXT_PROMPTS
)

INFO_OUTPUT_FILE = (
    TOTAL_IMAGES_INFO.parent
    / (
        f"total_{prompt_string}"
        f"_Sam3_auto_labels_save_info.txt"
    )
)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()