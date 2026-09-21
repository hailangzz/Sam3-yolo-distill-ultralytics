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
"""

import traceback
from pathlib import Path

from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor



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
        # "save": True,
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

def warmup_predictor(predictor):
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
            f"图片路径信息文件不存在：{TOTAL_IMAGES_INFO}"
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

            image_paths.append(Path(image_path))

    # 去重，同时保持原始顺序
    image_paths = list(dict.fromkeys(image_paths))

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
            # 这里兼容新的 info 格式：
            #
            # /xxx/xxx.txt:person,rug or carpet
            #
            # 判断是否已经存在时，只取 ":" 前面的 Label 路径
            # ------------------------------------------------

            label_path = path.split(":", 1)[0].strip()

            existing_paths.add(label_path)

    return existing_paths


# ============================================================
# 获取 Label 路径
# ============================================================

def get_label_path(image_path):
    """
    根据图片路径生成 Label 路径。

    所有 Label 统一保存到：

        Sam3_auto_labels/

    例如：

        xxx/1789439120669_xxx.jpg

    转换成：

        Sam3_auto_labels/
        1789439120669_xxx.txt
    """

    LABEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return LABEL_OUTPUT_DIR / f"{image_path.stem}.txt"


# ============================================================
# 获取图片尺寸
# ============================================================

def get_image_size(image_path, result=None):
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
            height, width = orig_shape[:2]

            return int(width), int(height)

    with Image.open(image_path) as image:
        width, height = image.size

    return width, height


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

    坐标归一化到：

        0 ~ 1
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

    # Mask polygon
    polygons = getattr(
        masks,
        "xy",
        None
    )

    # Class ID
    classes = getattr(
        boxes,
        "cls",
        None
    )

    if polygons is None:
        return labels

    if classes is None:
        return labels

    # 转 CPU / numpy
    try:
        class_ids = classes.cpu().numpy().astype(int)
    except Exception:
        class_ids = classes.numpy().astype(int)

    if len(polygons) == 0:
        return labels

    for polygon, class_id in zip(
        polygons,
        class_ids
    ):

        if polygon is None:
            continue

        if len(polygon) < 3:
            continue

        line = [str(class_id)]

        for point in polygon:

            x = float(point[0])
            y = float(point[1])

            # 归一化
            x_norm = x / image_width
            y_norm = y / image_height

            # 防止浮点误差超出 0~1
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

def get_detected_prompts(result):
    """
    获取当前 SAM3 Result 中实际检测到的 Prompt。

    SAM3 的 boxes.cls 中保存的是类别 ID：

        0 -> TEXT_PROMPTS[0]
        1 -> TEXT_PROMPTS[1]
        2 -> TEXT_PROMPTS[2]
        ...

    返回：
        当前图片实际检测到的 Prompt 字符串列表。

    例如：

        [
            "person",
            "rug or carpet"
        ]

    如果没有检测到目标：

        []
    """

    detected_prompts = []

    if result is None:
        return detected_prompts

    boxes = getattr(
        result,
        "boxes",
        None
    )

    if boxes is None:
        return detected_prompts

    classes = getattr(
        boxes,
        "cls",
        None
    )

    if classes is None:
        return detected_prompts

    try:
        class_ids = classes.cpu().numpy().astype(int)
    except Exception:
        class_ids = classes.numpy().astype(int)

    # --------------------------------------------------------
    # 去重，同时保持 TEXT_PROMPTS 中的类别顺序
    # --------------------------------------------------------

    detected_class_ids = set(
        int(class_id)
        for class_id in class_ids
    )

    for class_id, prompt in enumerate(
        TEXT_PROMPTS
    ):

        if class_id in detected_class_ids:

            detected_prompts.append(
                prompt
            )

    return detected_prompts


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

    try:

        if not image_path.exists():

            print(
                f"[WARNING] 图片不存在：{image_path}"
            )

            return None, [], "failed"

        label_path = get_label_path(
            image_path
        )

        # ----------------------------------------------------
        # 已经存在 Label
        # ----------------------------------------------------

        if SKIP_EXISTING_LABEL:

            if label_path.exists():

                print(
                    f"[SKIP] Label 已存在：{label_path}"
                )

                return label_path, [], "skip"

            if str(label_path) in existing_label_paths:

                print(
                    f"[SKIP] Info 中已存在记录：{label_path}"
                )

                return label_path, [], "skip"

        # ----------------------------------------------------
        # SAM3 推理
        # ----------------------------------------------------

        print(
            f"\n[PROCESS] {image_path}"
        )

        predictor.set_image(
            str(image_path)
        )

        results = predictor(
            text=TEXT_PROMPTS
        )

        # ----------------------------------------------------
        # 检查结果
        # ----------------------------------------------------

        if results is None:

            print(
                "[INFO] SAM3 没有返回结果"
            )

            print(
                "[INFO] 不生成 Label 文件"
            )

            return None, [], "no_target"

        # 通常这里是 list
        if not isinstance(
            results,
            (list, tuple)
        ):

            results = [results]

        all_labels = []

        # ----------------------------------------------------
        # 保存当前图片检测到的 Prompt
        # ----------------------------------------------------

        all_detected_prompts = []

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

            all_labels.extend(labels)

            # ------------------------------------------------
            # 获取检测到的 Prompt
            # ------------------------------------------------

            detected_prompts = (
                get_detected_prompts(result)
            )

            for prompt in detected_prompts:

                if prompt not in all_detected_prompts:

                    all_detected_prompts.append(
                        prompt
                    )

        # ----------------------------------------------------
        # 没有检测到目标
        # ----------------------------------------------------

        if not all_labels:

            print(
                f"[INFO] 未检测到目标：{image_path}"
            )

            print(
                "[INFO] 不生成 Label 文件"
            )

            return None, [], "no_target"

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
            f"[SUCCESS] Label 已生成：{label_path}"
        )

        print(
            f"[SUCCESS] 检测目标数量：{len(all_labels)}"
        )

        print(
            f"[SUCCESS] 检测到的 Prompt："
            f"{all_detected_prompts}"
        )

        return (
            label_path,
            all_detected_prompts,
            "success"
        )

    except Exception as e:

        print(
            f"\n[ERROR] 处理失败：{image_path}"
        )

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        # 打印完整 traceback
        traceback.print_exc()

        return None, [], "failed"


# ============================================================
# 主函数
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("SAM3 批量自动标注")
    print("=" * 80)

    print(
        f"[INFO] 图片列表：{TOTAL_IMAGES_INFO}"
    )

    print(
        f"[INFO] Prompt：{TEXT_PROMPTS}"
    )

    print(
        f"[INFO] Label 目录：{LABEL_OUTPUT_DIR}"
    )

    print(
        f"[INFO] Info 文件：{INFO_OUTPUT_FILE}"
    )

    print(
        f"[INFO] SKIP_EXISTING_LABEL："
        f"{SKIP_EXISTING_LABEL}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 读取图片路径
    # --------------------------------------------------------

    image_paths = load_image_paths()

    total_count = len(image_paths)

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
    #
    # 使用 append 模式。
    # 每成功生成一个 Label 就立即写入。
    # 即使中途程序异常退出，前面已经完成的数据
    # 也不会全部丢失。
    # --------------------------------------------------------

    with INFO_OUTPUT_FILE.open(
        "a",
        encoding="utf-8"
    ) as info_file:

        # ----------------------------------------------------
        # 顺序处理图片
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
            # 单张图片处理
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 成功
            # ------------------------------------------------

            if status == "success":

                success_count += 1

                # ------------------------------------------------
                # 生成 Info 记录
                #
                # 格式：
                #
                # /xxx/xxx.txt:person,rug or carpet
                #
                # ------------------------------------------------

                prompt_info = ",".join(
                    detected_prompts
                )

                info_file.write(
                    f"{label_path}:{prompt_info}\n"
                )

                # 立即 flush
                info_file.flush()

                existing_label_paths.add(
                    str(label_path)
                )

            # ------------------------------------------------
            # 已存在，跳过
            # ------------------------------------------------

            elif status == "skip":

                skip_count += 1

            # ------------------------------------------------
            # 没有目标
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
        f"[RESULT] 总图片数：      {total_count}"
    )

    print(
        f"[RESULT] 成功生成 Label： {success_count}"
    )

    print(
        f"[RESULT] 已存在跳过：     {skip_count}"
    )

    print(
        f"[RESULT] 无检测目标：     {no_target_count}"
    )

    print(
        f"[RESULT] 处理失败：       {failed_count}"
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

# 文本 Prompt
TEXT_PROMPTS = [
    "person",
    "rug or carpet",
    "Cables or wires on the ground",
    "Plastic sheets or plastic bags on the ground"
]

# 是否跳过已经存在的 Label
SKIP_EXISTING_LABEL = True

# SAM3 confidence
CONF = 0.25


# ============================================================
# 输出路径
# ============================================================

# Label 保存目录
LABEL_OUTPUT_DIR = (
    TOTAL_IMAGES_INFO.parent / "Sam3_auto_labels"
)

# 根据 Prompt 自动生成 info 文件名
prompt_string = "_".join(TEXT_PROMPTS)

INFO_OUTPUT_FILE = (
    TOTAL_IMAGES_INFO.parent
    / f"total_{prompt_string}_Sam3_auto_labels_save_info.txt"
)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
