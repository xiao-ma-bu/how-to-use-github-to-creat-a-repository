from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(r"F:\BasketballVideoMAE")

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "videomae_baseline"
    #/ "videomae_twostage"
    #/ "stage2_finetune"
    #/ "checkpoint-336"
    / "best_model"
    )

# 修改为需要预测的视频
VIDEO_PATH = Path(
    r"F:\BasketballVideoMAE\demo\2.mp4"
)


# ============================================================
# 2. 中文类别名称
# ============================================================

CHINESE_LABELS = {
    "one_stage": "一段式",
    "one_point_five_stage": "一点五段式",
    "two_stage": "二段式",
}


# ============================================================
# 3. 判定阈值
# ============================================================

# 如果最高类别概率低于该值，
# 认为模型无法判断属于哪一种已知投篮类型
UNKNOWN_THRESHOLD = 0.45

# 如果最高概率低于该值，
# 但又高于 UNKNOWN_THRESHOLD，
# 则认为结果不确定
CONFIDENCE_THRESHOLD = 0.6

# 第一名与第二名概率至少需要有这么大的差距
# 否则说明模型在两个类别之间犹豫
MARGIN_THRESHOLD = 0.3


# ============================================================
# 4. 读取并均匀采样视频
# ============================================================

def read_all_frames(
    path: Path,
) -> list[np.ndarray]:

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频：{path}"
        )

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            "视频中没有可读取帧。"
        )

    return frames

def multi_clip_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    num_clips: int = 5,
) -> list[list[np.ndarray]]:
    """
    从同一个视频生成多个时间采样版本。

    每个 clip 均覆盖完整视频，
    但在每个时间段中选择不同的位置。
    """

    total_frames = len(frames)

    if total_frames <= 0:
        raise RuntimeError(
            "视频没有有效帧。"
        )

    # 视频太短
    if total_frames < num_frames:

        indices = np.linspace(
            0,
            total_frames - 1,
            num=num_frames,
        ).round().astype(int)

        clip = [
            frames[index]
            for index in indices
        ]

        # 短视频没有太多时间采样空间
        return [clip]

    boundaries = np.linspace(
        0,
        total_frames,
        num=num_frames + 1,
    )

    # 例如5个clip：
    #
    # 0.1
    # 0.3
    # 0.5
    # 0.7
    # 0.9
    #
    relative_positions = np.linspace(
        0.1,
        0.9,
        num=num_clips,
    )

    clips = []

    for relative_position in relative_positions:

        sampled_indices = []

        for i in range(num_frames):

            start = int(
                np.floor(
                    boundaries[i]
                )
            )

            end = int(
                np.floor(
                    boundaries[i + 1]
                )
            )

            end = max(
                end,
                start + 1,
            )

            end = min(
                end,
                total_frames,
            )

            length = (
                end - start
            )

            index = (
                start
                + int(
                    relative_position
                    * max(length - 1, 0)
                )
            )

            index = min(
                index,
                total_frames - 1,
            )

            sampled_indices.append(
                index
            )

        clip = [
            frames[index]
            for index in sampled_indices
        ]

        clips.append(clip)

    return clips


# ============================================================
# 5. 主程序
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # 检查测试视频
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"测试视频不存在：{VIDEO_PATH}"
        )


    # --------------------------------------------------------
    # 选择运行设备
    # --------------------------------------------------------

    device = torch.device(
        # 如果只想使用 CPU：
        #"cpu"
       "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("篮球投篮方式识别")
    print("=" * 60)

    print(f"设备：{device}")
    print(f"视频：{VIDEO_PATH}")


    # --------------------------------------------------------
    # 加载处理器
    # --------------------------------------------------------

    processor = (
        VideoMAEImageProcessor.from_pretrained(
            MODEL_DIR
        )
    )


    # --------------------------------------------------------
    # 加载训练好的 VideoMAE 模型
    # --------------------------------------------------------

    model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            MODEL_DIR
        )
        .to(device)
    )

    model.eval()


    # --------------------------------------------------------
    # 读取视频
    # --------------------------------------------------------

    all_frames = read_all_frames(
        VIDEO_PATH
    )

    clips = multi_clip_sample_frames(
        frames=all_frames,
        num_frames=int(
            model.config.num_frames
        ),
        num_clips=5,
    )

    clip_probabilities = []

    with torch.no_grad():

        for clip in clips:
            inputs = processor(
                clip,
                return_tensors="pt",
            )

            pixel_values = (
                inputs["pixel_values"]
                .to(device)
            )

            outputs = model(
                pixel_values=pixel_values
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

            clip_probabilities.append(
                probabilities
            )

    # ============================================================
    # 输出每个 Clip 单独的预测结果
    # ============================================================

    print("\n" + "=" * 60)
    print("各时间采样 Clip 的预测结果")
    print("=" * 60)

    for clip_index, clip_probability in enumerate(
            clip_probabilities,
            start=1,
    ):

        print(
            f"\nClip {clip_index}"
        )

        # 按概率从高到低排序
        clip_sorted_indices = torch.argsort(
            clip_probability,
            descending=True,
        )

        for index in clip_sorted_indices:
            class_id = int(
                index.item()
            )

            label = (
                model.config.id2label[
                    class_id
                ]
            )

            score = float(
                clip_probability[
                    class_id
                ].item()
            )

            print(
                f"  "
                f"{CHINESE_LABELS.get(label, label)}："
                f"{score:.4f}"
            )

    # ============================================================
    # 多个时间采样结果取平均
    # ============================================================

    probabilities = torch.stack(
        clip_probabilities,
        dim=0,
    ).mean(dim=0)


    # --------------------------------------------------------
    # 按概率从高到低排序
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        probabilities,
        descending=True,
    )


    # --------------------------------------------------------
    # 输出三个类别概率
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("5 个 Clip 概率平均后的最终结果")
    print("=" * 60)

    for index in sorted_indices:
        class_id = int(
            index.item()
        )

        label = (
            model.config.id2label[
                class_id
            ]
        )

        score = float(
            probabilities[
                class_id
            ].item()
        )

        print(
            f"{CHINESE_LABELS.get(label, label)}："
            f"{score:.4f}"
        )


    # --------------------------------------------------------
    # 获取第一名
    # --------------------------------------------------------

    best_id = int(
        sorted_indices[0].item()
    )

    best_label = (
        model.config.id2label[
            best_id
        ]
    )

    best_score = float(
        probabilities[
            best_id
        ].item()
    )


    # --------------------------------------------------------
    # 获取第二名
    # --------------------------------------------------------

    second_id = int(
        sorted_indices[1].item()
    )

    second_label = (
        model.config.id2label[
            second_id
        ]
    )

    second_score = float(
        probabilities[
            second_id
        ].item()
    )


    # --------------------------------------------------------
    # 第一名与第二名概率差
    # --------------------------------------------------------

    confidence_margin = (
        best_score - second_score
    )


    print("\n" + "-" * 60)
    print("识别结果")
    print("-" * 60)

    print(
        f"最高置信度："
        f"{best_score:.4f}"
    )

    print(
        f"第二高置信度："
        f"{second_score:.4f}"
    )

    print(
        f"第一、第二类别差值："
        f"{confidence_margin:.4f}"
    )


    # ========================================================
    # 6. 最终判断
    # ========================================================

    # --------------------------------------------------------
    # 情况1：
    # 三种已知投篮方式都没有足够把握
    # --------------------------------------------------------

    if best_score < UNKNOWN_THRESHOLD:

        print("\n最终结果：无法识别投篮方式")

        print(
            "原因：模型对三种已知投篮方式"
            "均没有足够高的置信度。"
        )


    # --------------------------------------------------------
    # 情况2：
    # 有一定倾向，但置信度仍然不足
    # --------------------------------------------------------

    elif best_score < CONFIDENCE_THRESHOLD:

        print("\n最终结果：不确定")

        print(
            "可能类别："
            f"{CHINESE_LABELS.get(best_label, best_label)}"
        )

        print(
            f"置信度：{best_score:.4f}"
        )

        print(
            "原因：最高类别置信度不足。"
        )


    # --------------------------------------------------------
    # 情况3：
    # 第一名和第二名非常接近
    # --------------------------------------------------------

    elif confidence_margin < MARGIN_THRESHOLD:

        print("\n最终结果：不确定")

        print(
            "模型主要在以下两个类别之间犹豫："
        )

        print(
            f"1. "
            f"{CHINESE_LABELS.get(best_label, best_label)}"
            f"：{best_score:.4f}"
        )

        print(
            f"2. "
            f"{CHINESE_LABELS.get(second_label, second_label)}"
            f"：{second_score:.4f}"
        )

        print(
            "原因：第一名与第二名预测概率"
            "过于接近。"
        )


    # --------------------------------------------------------
    # 情况4：
    # 可以正常识别
    # --------------------------------------------------------

    else:

        print(
            "\n最终结果："
            f"{CHINESE_LABELS.get(best_label, best_label)}"
        )

        print(
            f"置信度："
            f"{best_score:.4f}"
        )


    print("=" * 60)


if __name__ == "__main__":
    main()