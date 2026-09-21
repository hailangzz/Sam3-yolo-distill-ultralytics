# -*- coding: utf-8 -*-

"""
SAM3 多线程批量图像推理

特点：
1. 给定一个图片目录
2. 多线程并行处理图片
3. 每个线程拥有独立的 SAM3SemanticPredictor
4. 每张图片只调用一次 set_image()
5. 支持多个 text prompt
6. 推理结果保存到独立目录
7. 实时显示处理速度
8. 支持失败重试/异常捕获
9. 支持递归搜索图片
10. 可通过 --workers 控制并发数

示例：

python sam3_multithread_infer.py \
    --input /data/database/jrdb_yolo_random_val/images/val \
    --output /data/sam3_results \
    --workers 2 \
    --device 0 \
    --prompts "carpet" "wire" "liquid" "plasticbag"

注意：
SAM3 每个 Worker 都会创建一个独立 Predictor。
因此 workers 越大，GPU 显存占用越高。

建议：
RTX 4060 Ti 8GB：
    先测试 workers=1
    再测试 workers=2
不要直接设置 4、8。
"""

import os
import cv2
import time
import json
import queue
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ultralytics.models.sam import SAM3SemanticPredictor


# ============================================================
# Worker-local Predictor
# ============================================================

_thread_local = threading.local()


def create_predictor(device: str):
    """
    为当前线程创建独立的 SAM3 Predictor。
    每个线程只初始化一次。
    """

    overrides = {
        "conf": 0.25,
        "task": "segment",
        "mode": "predict",
        "model": "sam3.pt",

        "save": False,

        # 指定 GPU
        "device": device,
    }

    print(
        f"[Worker-{threading.current_thread().name}] "
        f"Initializing SAM3..."
    )

    predictor = SAM3SemanticPredictor(
        overrides=overrides
    )

    print(
        f"[Worker-{threading.current_thread().name}] "
        f"SAM3 initialized."
    )

    return predictor


def get_predictor(device: str):
    """
    获取当前线程的 Predictor。
    """

    if not hasattr(_thread_local, "predictor"):

        _thread_local.predictor = create_predictor(
            device
        )

    return _thread_local.predictor


# ============================================================
# Result serialization
# ============================================================

def save_results(results, image_path: Path, output_dir: Path):
    """
    保存 SAM3 推理结果。

    保存内容：

    output/
        image.jpg
        image.json

    JSON 中保存：
        prompt
        confidence
        bbox
        polygon
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = output_dir / (
        image_path.stem + ".json"
    )

    all_objects = []

    for result in results:

        # ----------------------------------------------------
        # Boxes
        # ----------------------------------------------------

        boxes = None

        if result.boxes is not None:
            boxes = result.boxes

        # ----------------------------------------------------
        # Masks
        # ----------------------------------------------------

        masks = result.masks

        if boxes is None:
            continue

        num_objects = len(boxes)

        for i in range(num_objects):

            obj = {}

            # class
            if result.names is not None:
                cls_id = int(
                    boxes.cls[i].item()
                )

                obj["class_id"] = cls_id
                obj["class_name"] = result.names.get(
                    cls_id,
                    str(cls_id)
                )

            # confidence
            if boxes.conf is not None:
                obj["confidence"] = float(
                    boxes.conf[i].item()
                )

            # bbox
            if boxes.xyxy is not None:
                obj["bbox"] = [
                    float(x)
                    for x in boxes.xyxy[i].tolist()
                ]

            # mask polygon
            if masks is not None:

                if hasattr(masks, "xy"):

                    polygons = masks.xy

                    if i < len(polygons):

                        polygon = polygons[i]

                        obj["polygon"] = [
                            [
                                float(x),
                                float(y)
                            ]
                            for x, y in polygon
                        ]

            all_objects.append(obj)

    data = {
        "image": str(image_path),
        "objects": all_objects
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# Single image inference
# ============================================================

def process_one_image(
    image_path: Path,
    output_dir: Path,
    prompts,
    device: str,
):

    start_time = time.perf_counter()

    try:

        predictor = get_predictor(device)

        # ----------------------------------------------------
        # 设置图片
        # ----------------------------------------------------

        predictor.set_image(
            str(image_path)
        )

        # ----------------------------------------------------
        # SAM3 inference
        # ----------------------------------------------------

        results = predictor(
            text=prompts
        )

        # ----------------------------------------------------
        # 保存结果
        # ----------------------------------------------------

        save_results(
            results,
            image_path,
            output_dir
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "success": True,
            "image": str(image_path),
            "time": elapsed,
        }

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "success": False,
            "image": str(image_path),
            "time": elapsed,
            "error": repr(e),
        }


# ============================================================
# Find images
# ============================================================

def find_images(input_dir: Path):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = []

    for path in input_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in extensions:
            images.append(path)

    images.sort()

    return images


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="图片目录"
    )

    parser.add_argument(
        "--output",
        default="./sam3_results",
        help="结果目录"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="SAM3 并发 Worker 数量"
    )

    parser.add_argument(
        "--device",
        default="0",
        help="GPU，例如 0"
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="SAM3 text prompts"
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    # --------------------------------------------------------
    # 检查目录
    # --------------------------------------------------------

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    # --------------------------------------------------------
    # 找图片
    # --------------------------------------------------------

    images = find_images(
        input_dir
    )

    if len(images) == 0:

        print(
            f"No images found in {input_dir}"
        )

        return

    print()
    print("=" * 70)
    print("SAM3 Multi-thread Inference")
    print("=" * 70)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Images:      {len(images)}")
    print(f"Workers:     {args.workers}")
    print(f"Device:      {args.device}")
    print(f"Prompts:     {args.prompts}")
    print("=" * 70)
    print()

    total_start = time.perf_counter()

    success_count = 0
    failed_count = 0

    completed = 0

    # --------------------------------------------------------
    # ThreadPool
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="SAM3"
    ) as executor:

        futures = []

        for image_path in images:

            future = executor.submit(
                process_one_image,
                image_path,
                output_dir,
                args.prompts,
                args.device,
            )

            futures.append(future)

        # ----------------------------------------------------
        # Collect result
        # ----------------------------------------------------

        for future in as_completed(futures):

            result = future.result()

            completed += 1

            if result["success"]:

                success_count += 1

                print(
                    f"[{completed:5d}/{len(images)}] "
                    f"OK   "
                    f"{Path(result['image']).name:<40} "
                    f"{result['time']:.2f}s"
                )

            else:

                failed_count += 1

                print(
                    f"[{completed:5d}/{len(images)}] "
                    f"FAIL "
                    f"{Path(result['image']).name:<40} "
                    f"{result['error']}"
                )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    throughput = (
        len(images) / total_time
        if total_time > 0
        else 0
    )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)

    print(
        f"Total images : {len(images)}"
    )

    print(
        f"Success      : {success_count}"
    )

    print(
        f"Failed       : {failed_count}"
    )

    print(
        f"Total time   : {total_time:.2f}s"
    )

    print(
        f"Throughput   : {throughput:.3f} images/s"
    )

    print(
        f"Average      : "
        f"{total_time / len(images):.3f}s/image"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

