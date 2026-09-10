
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8-Seg 蒸馏模型推理测试程序

模型：
    /data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/checkpoints/best.pt

功能：
    1. 加载蒸馏训练后的 YOLOv8-Seg 模型
    2. 支持单张图片
    3. 支持整个图片目录
    4. 图片按文件名升序排列
    5. A：上一张
    6. D：下一张
    7. Q / ESC：退出
    8. 支持置信度阈值
    9. 支持 IoU 阈值
    10. 支持指定类别显示
    11. 显示分割 Mask
    12. 显示类别名称和置信度
    13. 支持保存推理结果
    14. 显示单张推理耗时

运行示例：

    python inference_test.py \
        --source /path/to/images

或者：

    python inference_test.py \
        --source /path/to/image.jpg

指定置信度：

    python inference_test.py \
        --source /path/to/images \
        --conf 0.5

只显示 carpet 和 wire：

    python inference_test.py \
        --source /path/to/images \
        --classes 0 1

保存结果：

    python inference_test.py \
        --source /data/Sam3-yolo-distill-ultralytics/tests/test.jpg \
        --save-dir inference_results


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
    "/data/Sam3-yolo-distill-ultralytics/Sam3-yolo-distill/checkpoints/distilled_yolov8n-seg.pt"
)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640


# ============================================================
# 类别定义
# ============================================================

CLASS_NAMES = {
    0: "carpet",
    1: "wire",
    2: "liquid",
    3: "plastic bag",
}


# ============================================================
# 获取图片
# ============================================================

def collect_images(source):
    """
    source 可以是：

        1. 单张图片
        2. 图片目录
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

    if source.is_file():

        if source.suffix.lower() not in image_extensions:
            raise ValueError(
                f"不支持的图片格式：{source.suffix}"
            )

        return [source]

    if source.is_dir():

        images = [
            p
            for p in source.iterdir()
            if p.is_file()
            and p.suffix.lower() in image_extensions
        ]

        # 文件名升序
        images.sort(key=lambda x: x.name.lower())

        return images

    raise FileNotFoundError(
        f"找不到 source：{source}"
    )


# ============================================================
# 获取类别名称
# ============================================================

def get_class_name(model, class_id):
    """
    优先使用模型自身 names。

    如果模型 names 不存在，
    使用当前项目的 CLASS_NAMES。
    """

    try:

        names = model.names

        if isinstance(names, dict):

            return names.get(
                class_id,
                CLASS_NAMES.get(class_id, str(class_id))
            )

        if isinstance(names, list):

            if 0 <= class_id < len(names):
                return names[class_id]

    except Exception:
        pass

    return CLASS_NAMES.get(
        class_id,
        str(class_id)
    )


# ============================================================
# 绘制 Mask
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
        None -> 显示所有类别

        [0, 1]
            -> 只显示 carpet / wire
    """

    output = image.copy()

    if result.masks is None:
        return output, 0

    if result.boxes is None:
        return output, 0

    masks = result.masks.data
    boxes = result.boxes

    count = 0

    for i in range(len(boxes)):

        cls_id = int(
            boxes.cls[i].item()
        )

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

        mask = masks[i].cpu().numpy()

        # resize 到原图尺寸
        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_bool = mask > 0.5

        # ----------------------------------------------------
        # 生成随机但稳定的类别颜色
        # ----------------------------------------------------

        rng = np.random.default_rng(
            cls_id + 12345
        )

        color = rng.integers(
            0,
            256,
            size=3,
        ).tolist()

        color = tuple(
            int(x)
            for x in color
        )

        # ----------------------------------------------------
        # 绘制半透明 Mask
        # ----------------------------------------------------

        overlay = output.copy()

        overlay[mask_bool] = color

        output = cv2.addWeighted(
            overlay,
            0.45,
            output,
            0.55,
            0,
        )

        # ----------------------------------------------------
        # 绘制轮廓
        # ----------------------------------------------------

        mask_uint8 = (
            mask_bool.astype(np.uint8) * 255
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

        xyxy = boxes.xyxy[i].cpu().numpy()

        x1, y1, x2, y2 = map(
            int,
            xyxy
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
            cls_id
        )

        label = (
            f"{class_name} "
            f"{conf:.2f}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), baseline = cv2.getTextSize(
            label,
            font,
            0.6,
            2,
        )

        label_y = max(
            y1,
            th + baseline + 2
        )

        cv2.rectangle(
            output,
            (
                x1,
                label_y - th - baseline - 4
            ),
            (
                x1 + tw + 4,
                label_y
            ),
            color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                x1 + 2,
                label_y - baseline - 2
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

    info1 = (
        f"[{index + 1}/{total}] "
        f"{image_path.name}"
    )

    info2 = (
        f"inference: {inference_time:.2f} ms    "
        f"objects: {object_count}"
    )

    info3 = (
        "A: previous    "
        "D: next    "
        "Q / ESC: quit"
    )

    # 半透明黑色背景
    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 82),
        (0, 0, 0),
        -1,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

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
# 推理单张图片
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
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"无法读取图片：{image_path}"
        )

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
        time.perf_counter() - start
    ) * 1000.0

    result = results[0]

    output, count = draw_result(
        image,
        result,
        model,
        selected_classes,
    )

    output = draw_info(
        output,
        0,
        1,
        image_path,
        elapsed,
        count,
    )

    return output, elapsed, count, result


# ============================================================
# 保存结果
# ============================================================

def save_result(
    image,
    image_path,
    save_dir,
):
    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        save_dir / image_path.name
    )

    cv2.imwrite(
        str(output_path),
        image,
    )

    return output_path


# ============================================================
# 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "YOLOv8-Seg 蒸馏模型推理测试程序"
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
        default=None,
        help="保存推理结果目录",
    )

    args = parser.parse_args()

    # ========================================================
    # 检查模型
    # ========================================================

    if not os.path.exists(args.model):

        raise FileNotFoundError(
            f"模型不存在：\n{args.model}"
        )

    print("=" * 70)
    print("YOLOv8-Seg 蒸馏模型推理测试")
    print("=" * 70)

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

    print("=" * 70)

    # ========================================================
    # 加载模型
    # ========================================================

    print("\n正在加载模型...")

    model = YOLO(
        args.model
    )

    print("模型加载成功。")

    print(
        f"Model classes: {model.names}"
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

    print(
        f"\n共找到 {len(images)} 张图片。"
    )

    # ========================================================
    # OpenCV 窗口
    # ========================================================

    window_name = (
        "YOLOv8-Seg Distillation Inference"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    # 窗口大小可以根据需要调整
    cv2.resizeWindow(
        window_name,
        1280,
        900,
    )

    # ========================================================
    # 当前图片
    # ========================================================

    index = 0

    total_inference_time = 0.0
    inference_count = 0

    while True:

        image_path = images[index]

        # ----------------------------------------------------
        # 读取图片
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"无法读取：{image_path}"
            )

            index = (
                index + 1
            ) % len(images)

            continue

        # ----------------------------------------------------
        # 推理
        # ----------------------------------------------------

        start = time.perf_counter()

        results = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        result = results[0]

        # ----------------------------------------------------
        # 绘制
        # ----------------------------------------------------

        output, object_count = draw_result(
            image,
            result,
            model,
            args.classes,
        )

        output = draw_info(
            output,
            index,
            len(images),
            image_path,
            elapsed,
            object_count,
        )

        # ----------------------------------------------------
        # 显示
        # ----------------------------------------------------

        cv2.imshow(
            window_name,
            output,
        )

        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        if args.save_dir is not None:

            save_path = save_result(
                output,
                image_path,
                args.save_dir,
            )

            print(
                f"[SAVE] {save_path}"
            )

        # ----------------------------------------------------
        # 统计
        # ----------------------------------------------------

        total_inference_time += elapsed
        inference_count += 1

        # ----------------------------------------------------
        # 键盘
        # ----------------------------------------------------

        key = cv2.waitKey(0) & 0xFF

        # ESC / Q
        if key == 27 or key in (
            ord("q"),
            ord("Q"),
        ):
            break

        # A：上一张
        elif key in (
            ord("a"),
            ord("A"),
        ):

            index -= 1

            if index < 0:
                index = (
                    len(images) - 1
                )

        # D：下一张
        elif key in (
            ord("d"),
            ord("D"),
        ):

            index += 1

            if index >= len(images):
                index = 0

    # ========================================================
    # 清理
    # ========================================================

    cv2.destroyAllWindows()

    # ========================================================
    # 统计信息
    # ========================================================

    print("\n" + "=" * 70)
    print("推理测试结束")
    print("=" * 70)

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

        print(
            f"测试图片数：{inference_count}"
        )

        print(
            f"平均推理时间：{avg_time:.2f} ms"
        )

        print(
            f"理论 FPS：{fps:.2f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()

