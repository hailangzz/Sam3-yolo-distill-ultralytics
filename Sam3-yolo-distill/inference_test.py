
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8-Seg 蒸馏模型推理测试程序

============================================================
功能
============================================================

1. 加载标准 YOLOv8-Seg 蒸馏模型
2. 支持单张图片
3. 支持整个图片目录
4. 图片按照文件名升序排列
5. A：上一张
6. D：下一张
7. Q / ESC：退出
8. 支持置信度阈值
9. 支持 IoU 阈值
10. 支持指定类别显示
11. 显示分割 Mask
12. 显示 Mask 轮廓
13. 显示 Bounding Box
14. 显示类别名称和置信度
15. 自动保存推理结果图片
16. 显示单张推理耗时
17. 显示平均推理耗时
18. 显示理论 FPS
19. 支持 --no-display 无 GUI 模式
20. 无 GUI 模式下可批量处理整个目录

============================================================
默认模型
============================================================

/data/Sam3-yolo-distill-ultralytics/
Sam3-yolo-distill/checkpoints/
distilled_yolov8n-seg.pt

============================================================
类别
============================================================

0: carpet
1: wire
2: liquid
3: plasticbag

============================================================
运行示例
============================================================

单张图片：

python inference_test.py \
    --source /path/to/image.jpg

图片目录：

python inference_test.py \
    --source /path/to/images

指定置信度：

python inference_test.py \
    --source /path/to/images \
    --conf 0.5

只显示 carpet 和 wire：

python inference_test.py \
    --source /path/to/images \
    --classes 0 1

指定保存目录：

python inference_test.py \
    --source /path/to/images \
    --save-dir inference_results

无 GUI 批量推理：

python inference_test.py \
    --source /path/to/images \
    --save-dir inference_results \
    --no-display

============================================================
"""

import os
import cv2
import time
import argparse
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 默认配置
# ============================================================

DEFAULT_MODEL = (
    "/data/Sam3-yolo-distill-ultralytics/"
    "Sam3-yolo-distill/checkpoints/"
    "distilled_yolov8n-seg.pt"
)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

DEFAULT_SAVE_DIR = "inference_results"


# ============================================================
# 类别定义
# ============================================================

CLASS_NAMES = {
    0: "carpet",
    1: "wire",
    2: "liquid",
    3: "plasticbag",
}


# ============================================================
# 类别颜色
# ============================================================

def get_class_color(class_id):
    """
    根据类别 ID 生成稳定颜色。

    同一个类别每次运行都会得到相同颜色。
    """

    rng = np.random.default_rng(
        class_id + 12345
    )

    color = rng.integers(
        0,
        256,
        size=3,
    ).tolist()

    return tuple(
        int(x)
        for x in color
    )


# ============================================================
# 获取图片
# ============================================================

def collect_images(source):
    """
    source 可以是：

        1. 单张图片
        2. 图片目录

    返回：
        List[Path]

    图片按照文件名升序排列。
    """

    source = Path(source)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    # --------------------------------------------------------
    # 单张图片
    # --------------------------------------------------------

    if source.is_file():

        if (
            source.suffix.lower()
            not in image_extensions
        ):

            raise ValueError(
                f"不支持的图片格式："
                f"{source.suffix}"
            )

        return [source]

    # --------------------------------------------------------
    # 图片目录
    # --------------------------------------------------------

    if source.is_dir():

        images = [
            p
            for p in source.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in image_extensions
            )
        ]

        # 文件名升序
        images.sort(
            key=lambda x: x.name.lower()
        )

        return images

    raise FileNotFoundError(
        f"找不到 source：{source}"
    )


# ============================================================
# 获取类别名称
# ============================================================

def get_class_name(
    model,
    class_id,
):
    """
    优先使用模型自身 names。

    如果模型 names 不存在，
    使用当前项目 CLASS_NAMES。
    """

    try:

        names = model.names

        if isinstance(
            names,
            dict
        ):

            return names.get(
                class_id,
                CLASS_NAMES.get(
                    class_id,
                    str(class_id),
                ),
            )

        if isinstance(
            names,
            list
        ):

            if (
                0 <= class_id
                < len(names)
            ):

                return names[class_id]

    except Exception:
        pass

    return CLASS_NAMES.get(
        class_id,
        str(class_id),
    )


# ============================================================
# 绘制 Mask / Box / Label
# ============================================================

def draw_result(
    image,
    result,
    model,
    selected_classes=None,
):
    """
    手动绘制 YOLO Segmentation 结果。

    selected_classes:

        None
            显示所有类别

        [0, 1]
            只显示 carpet / wire
    """

    output = image.copy()

    # --------------------------------------------------------
    # 没有 Mask
    # --------------------------------------------------------

    if result.masks is None:

        return output, 0

    # --------------------------------------------------------
    # 没有 Box
    # --------------------------------------------------------

    if result.boxes is None:

        return output, 0

    masks = result.masks.data

    boxes = result.boxes

    count = 0

    # ========================================================
    # 遍历所有检测结果
    # ========================================================

    for i in range(
        len(boxes)
    ):

        # ----------------------------------------------------
        # 类别
        # ----------------------------------------------------

        cls_id = int(
            boxes.cls[i].item()
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        conf = float(
            boxes.conf[i].item()
        )

        # ----------------------------------------------------
        # 类别过滤
        # ----------------------------------------------------

        if (
            selected_classes is not None
            and cls_id not in selected_classes
        ):

            continue

        # ----------------------------------------------------
        # 获取 Mask
        # ----------------------------------------------------

        mask = (
            masks[i]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # resize 到原图尺寸
        # ----------------------------------------------------

        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_bool = (
            mask > 0.5
        )

        # ----------------------------------------------------
        # 获取类别颜色
        # ----------------------------------------------------

        color = get_class_color(
            cls_id
        )

        # ----------------------------------------------------
        # 半透明 Mask
        # ----------------------------------------------------

        overlay = output.copy()

        overlay[
            mask_bool
        ] = color

        output = cv2.addWeighted(
            overlay,
            0.45,
            output,
            0.55,
            0,
        )

        # ----------------------------------------------------
        # Mask 轮廓
        # ----------------------------------------------------

        mask_uint8 = (
            mask_bool.astype(
                np.uint8
            )
            * 255
        )

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            output,
            contours,
            -1,
            color,
            2,
        )

        # ----------------------------------------------------
        # Bounding Box
        # ----------------------------------------------------

        xyxy = (
            boxes.xyxy[i]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = map(
            int,
            xyxy,
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        class_name = get_class_name(
            model,
            cls_id,
        )

        label = (
            f"{class_name} "
            f"{conf:.2f}"
        )

        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        (
            tw,
            th,
        ), baseline = cv2.getTextSize(
            label,
            font,
            0.6,
            2,
        )

        label_y = max(
            y1,
            th + baseline + 2,
        )

        # ----------------------------------------------------
        # Label 背景
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (
                x1,
                label_y
                - th
                - baseline
                - 4,
            ),
            (
                x1 + tw + 4,
                label_y,
            ),
            color,
            -1,
        )

        # ----------------------------------------------------
        # Label 文字
        # ----------------------------------------------------

        cv2.putText(
            output,
            label,
            (
                x1 + 2,
                label_y
                - baseline
                - 2,
            ),
            font,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        count += 1

    return output, count


# ============================================================
# 绘制顶部信息
# ============================================================

def draw_info(
    image,
    index,
    total,
    image_path,
    inference_time,
    object_count,
):
    """
    在图像顶部显示当前状态。
    """

    output = image.copy()

    # --------------------------------------------------------
    # 第一行
    # --------------------------------------------------------

    info1 = (
        f"[{index + 1}/{total}] "
        f"{image_path.name}"
    )

    # --------------------------------------------------------
    # 第二行
    # --------------------------------------------------------

    info2 = (
        f"inference: "
        f"{inference_time:.2f} ms    "
        f"objects: "
        f"{object_count}"
    )

    # --------------------------------------------------------
    # 第三行
    # --------------------------------------------------------

    info3 = (
        "A: previous    "
        "D: next    "
        "Q / ESC: quit"
    )

    # --------------------------------------------------------
    # 顶部背景
    # --------------------------------------------------------

    cv2.rectangle(
        output,
        (0, 0),
        (
            output.shape[1],
            82,
        ),
        (0, 0, 0),
        -1,
    )

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    # --------------------------------------------------------
    # 第一行
    # --------------------------------------------------------

    cv2.putText(
        output,
        info1,
        (10, 22),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # 第二行
    # --------------------------------------------------------

    cv2.putText(
        output,
        info2,
        (10, 47),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # 第三行
    # --------------------------------------------------------

    cv2.putText(
        output,
        info3,
        (10, 70),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# 保存结果
# ============================================================

def save_result(
    image,
    image_path,
    save_dir,
):
    """
    保存绘制后的最终结果图片。

    文件名保持和原图一致。
    """

    save_dir = Path(
        save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        save_dir
        / image_path.name
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:

        raise RuntimeError(
            f"保存图片失败："
            f"{output_path}"
        )

    return output_path


# ============================================================
# 单张图片推理
# ============================================================

def inference_image(
    model,
    image_path,
    conf,
    iou,
    imgsz,
    device,
    selected_classes,
):
    """
    对单张图片进行 YOLO 推理。

    返回：

        output
        elapsed
        object_count
        result
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"无法读取图片："
            f"{image_path}"
        )

    # --------------------------------------------------------
    # 推理开始
    # --------------------------------------------------------

    start = time.perf_counter()

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    result = results[0]

    # --------------------------------------------------------
    # 绘制
    # --------------------------------------------------------

    output, object_count = draw_result(
        image,
        result,
        model,
        selected_classes,
    )

    return (
        output,
        elapsed,
        object_count,
        result,
    )


# ============================================================
# 主程序
# ============================================================

def main():

    # ========================================================
    # 参数
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "YOLOv8-Seg "
            "蒸馏模型推理测试程序"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="模型路径",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="图片或图片目录",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="置信度阈值",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help="NMS IoU 阈值",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="推理图片尺寸",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="设备，例如 0 / cpu",
    )

    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "只显示指定类别，例如："
            "--classes 0 1"
        ),
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=(
            "保存推理结果目录"
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "关闭 OpenCV GUI。"
            "适用于服务器/无显示环境。"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # 参数检查
    # ========================================================

    if not (
        0.0
        <= args.conf
        <= 1.0
    ):

        raise ValueError(
            "--conf 必须在 0~1 之间"
        )

    if not (
        0.0
        <= args.iou
        <= 1.0
    ):

        raise ValueError(
            "--iou 必须在 0~1 之间"
        )

    # ========================================================
    # 检查模型
    # ========================================================

    if not os.path.exists(
        args.model
    ):

        raise FileNotFoundError(
            f"模型不存在：\n"
            f"{args.model}"
        )

    # ========================================================
    # 打印配置
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "YOLOv8-Seg 蒸馏模型推理测试"
    )

    print(
        "=" * 70
    )

    print(
        f"Model : {args.model}"
    )

    print(
        f"Source: {args.source}"
    )

    print(
        f"Conf  : {args.conf}"
    )

    print(
        f"IoU   : {args.iou}"
    )

    print(
        f"ImgSz : {args.imgsz}"
    )

    print(
        f"Device: {args.device}"
    )

    print(
        f"Classes: {args.classes}"
    )

    print(
        f"SaveDir: {args.save_dir}"
    )

    print(
        f"Display: "
        f"{'OFF' if args.no_display else 'ON'}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 加载模型
    # ========================================================

    print()
    print(
        "正在加载模型..."
    )

    model = YOLO(
        args.model
    )

    print(
        "模型加载成功。"
    )

    print(
        f"Model classes: "
        f"{model.names}"
    )

    # ========================================================
    # 验证类别
    # ========================================================

    expected_class_count = len(
        CLASS_NAMES
    )

    if len(model.names) != expected_class_count:

        raise RuntimeError(
            "模型类别数量异常：\n"
            f"Expected: "
            f"{expected_class_count}\n"
            f"Actual: "
            f"{len(model.names)}"
        )

    print()
    print(
        "模型类别验证通过："
    )

    for class_id, name in (
        model.names.items()
        if isinstance(
            model.names,
            dict
        )
        else enumerate(
            model.names
        )
    ):

        print(
            f"   {class_id}: {name}"
        )

    # ========================================================
    # 获取图片
    # ========================================================

    images = collect_images(
        args.source
    )

    if len(images) == 0:

        raise RuntimeError(
            "没有找到图片。"
        )

    print()
    print(
        f"共找到 {len(images)} 张图片。"
    )

    # ========================================================
    # 创建保存目录
    # ========================================================

    save_dir = Path(
        args.save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "推理结果将保存到："
    )

    print(
        f"   {save_dir.resolve()}"
    )

    # ========================================================
    # OpenCV GUI
    # ========================================================

    window_name = (
        "YOLOv8-Seg Distillation Inference"
    )

    display_enabled = (
        not args.no_display
    )

    if display_enabled:

        try:

            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                window_name,
                1280,
                900,
            )

            print()
            print(
                "OpenCV GUI 初始化成功。"
            )

        except Exception as e:

            print()
            print(
                "WARNING:"
            )

            print(
                "OpenCV GUI 初始化失败："
            )

            print(
                f"   {e}"
            )

            print()
            print(
                "自动切换到无 GUI 模式。"
            )

            display_enabled = False

    # ========================================================
    # 当前图片
    # ========================================================

    index = 0

    # ========================================================
    # 统计
    # ========================================================

    total_inference_time = 0.0

    inference_count = 0

    total_objects = 0

    # ========================================================
    # 推理循环
    # ========================================================

    while True:

        image_path = images[index]

        print()
        print(
            "-" * 70
        )

        print(
            f"[{index + 1}/{len(images)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # 推理
        # ----------------------------------------------------

        try:

            (
                output,
                elapsed,
                object_count,
                result,
            ) = inference_image(
                model,
                image_path,
                args.conf,
                args.iou,
                args.imgsz,
                args.device,
                args.classes,
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"推理失败：{image_path}"
            )

            print(
                f"       {e}"
            )

            # 进入下一张
            if len(images) == 1:
                break

            index += 1

            if (
                index
                >= len(images)
            ):

                index = 0

            continue

        # ----------------------------------------------------
        # 绘制顶部信息
        # ----------------------------------------------------

        output = draw_info(
            output,
            index,
            len(images),
            image_path,
            elapsed,
            object_count,
        )

        # ----------------------------------------------------
        # 保存结果
        # ----------------------------------------------------

        save_path = save_result(
            output,
            image_path,
            save_dir,
        )

        print(
            f"[SAVE] {save_path}"
        )

        # ----------------------------------------------------
        # 打印检测结果
        # ----------------------------------------------------

        if result.boxes is None:

            print(
                "Detections: 0"
            )

        else:

            print(
                f"Detections: "
                f"{len(result.boxes)}"
            )

            for i in range(
                len(result.boxes)
            ):

                cls_id = int(
                    result.boxes.cls[i].item()
                )

                conf = float(
                    result.boxes.conf[i].item()
                )

                class_name = (
                    get_class_name(
                        model,
                        cls_id,
                    )
                )

                print(
                    f"   "
                    f"{i}: "
                    f"class={cls_id} "
                    f"({class_name}) "
                    f"conf={conf:.4f}"
                )

        # ----------------------------------------------------
        # 推理耗时
        # ----------------------------------------------------

        print(
            f"Inference: "
            f"{elapsed:.2f} ms"
        )

        print(
            f"Objects: "
            f"{object_count}"
        )

        # ----------------------------------------------------
        # 累计统计
        # ----------------------------------------------------

        total_inference_time += (
            elapsed
        )

        inference_count += 1

        total_objects += (
            object_count
        )

        # ====================================================
        # 无 GUI 模式
        # ====================================================

        if not display_enabled:

            # ------------------------------------------------
            # 单张图片
            # ------------------------------------------------

            if len(images) == 1:

                break

            # ------------------------------------------------
            # 批量模式
            # ------------------------------------------------

            index += 1

            if (
                index
                >= len(images)
            ):

                break

            continue

        # ====================================================
        # GUI 显示
        # ====================================================

        cv2.imshow(
            window_name,
            output,
        )

        # ====================================================
        # 等待键盘
        # ====================================================

        key = (
            cv2.waitKey(0)
            & 0xFF
        )

        # ----------------------------------------------------
        # Q / ESC
        # ----------------------------------------------------

        if (
            key == 27
            or key in (
                ord("q"),
                ord("Q"),
            )
        ):

            break

        # ----------------------------------------------------
        # A：上一张
        # ----------------------------------------------------

        elif key in (
            ord("a"),
            ord("A"),
        ):

            index -= 1

            if index < 0:

                index = (
                    len(images) - 1
                )

        # ----------------------------------------------------
        # D：下一张
        # ----------------------------------------------------

        elif key in (
            ord("d"),
            ord("D"),
        ):

            index += 1

            if (
                index
                >= len(images)
            ):

                index = 0

    # ========================================================
    # 清理 GUI
    # ========================================================

    if display_enabled:

        try:

            cv2.destroyAllWindows()

        except Exception:
            pass

    # ========================================================
    # 最终统计
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "推理测试结束"
    )

    print(
        "=" * 70
    )

    if inference_count > 0:

        avg_time = (
            total_inference_time
            / inference_count
        )

        fps = (
            1000.0 / avg_time
            if avg_time > 0
            else 0
        )

        avg_objects = (
            total_objects
            / inference_count
        )

        print(
            f"测试图片数："
            f"{inference_count}"
        )

        print(
            f"平均推理时间："
            f"{avg_time:.2f} ms"
        )

        print(
            f"理论 FPS："
            f"{fps:.2f}"
        )

        print(
            f"平均目标数量："
            f"{avg_objects:.2f}"
        )

    print()
    print(
        f"结果目录："
        f"{save_dir.resolve()}"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()


"""

python inference_test.py \
    --source /data/Sam3-yolo-distill-ultralytics/tests/test.jpg \
    --save-dir inference_results \
    --no-display
    
    
"""