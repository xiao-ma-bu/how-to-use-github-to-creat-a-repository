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

# 与新的两阶段训练代码保持一致
MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "videomae_twostage"

    / "best_model"
)

# 修改为需要预测的视频
VIDEO_PATH = PROJECT_ROOT / "demo" / "2.mp4"


# ============================================================
# 2. 类别名称
# ============================================================
CHINESE_LABELS = {
    "one_stage": "一段式",
    "one_point_five_stage": "一点五段式",
    "two_stage": "二段式",
}


# ============================================================
# 3. Multi-Clip 与拒识阈值
# ============================================================
NUM_CLIPS = 5

# 注意：这三个阈值是经验值。
# Label Smoothing 会改变概率分布，因此后续最好根据 VAL 集重新标定。
UNKNOWN_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.30


# ============================================================
# 4. 视频读取
# ============================================================
def read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("视频中没有可读取帧")
    return frames


# ============================================================
# 5. 5-Clip时间采样
# ============================================================
def multi_clip_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    num_clips: int = 5,
) -> list[list[np.ndarray]]:
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total < num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        return [[frames[i] for i in indices]]

    boundaries = np.linspace(0, total, num=num_frames + 1)
    relative_positions = np.linspace(0.1, 0.9, num=num_clips)
    clips: list[list[np.ndarray]] = []

    for pos in relative_positions:
        indices: list[int] = []
        for i in range(num_frames):
            start = int(np.floor(boundaries[i]))
            end = int(np.floor(boundaries[i + 1]))
            end = max(end, start + 1)
            end = min(end, total)
            length = end - start
            index = start + int(pos * max(length - 1, 0))
            indices.append(min(index, total - 1))
        clips.append([frames[i] for i in indices])

    return clips


def get_model_label(
    model: VideoMAEForVideoClassification,
    class_id: int,
) -> str:
    """兼容 id2label 的 int/string key。"""
    mapping = model.config.id2label
    if class_id in mapping:
        return mapping[class_id]
    if str(class_id) in mapping:
        return mapping[str(class_id)]
    return str(class_id)


# ============================================================
# 6. 主程序
# ============================================================
def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"模型目录不存在：{MODEL_DIR}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"测试视频不存在：{VIDEO_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("篮球投篮方式识别 - Multi-Clip")
    print("=" * 60)
    print(f"设备：{device}")
    print(f"模型：{MODEL_DIR}")
    print(f"视频：{VIDEO_PATH}")
    print(f"目标Clip数量：{NUM_CLIPS}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    all_frames = read_all_frames(VIDEO_PATH)
    clips = multi_clip_sample_frames(
        all_frames,
        num_frames=int(model.config.num_frames),
        num_clips=NUM_CLIPS,
    )

    print(f"实际生成Clip数量：{len(clips)}")

    clip_probabilities: list[torch.Tensor] = []

    with torch.inference_mode():
        for clip in clips:
            inputs = processor(clip, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values).logits
            probability = torch.softmax(logits, dim=-1)[0].cpu()
            clip_probabilities.append(probability)

    # --------------------------------------------------------
    # 每个Clip单独结果
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("各时间Clip预测")
    print("=" * 60)

    for clip_index, probability in enumerate(clip_probabilities, start=1):
        print(f"\nClip {clip_index}")
        order = torch.argsort(probability, descending=True)
        for index in order:
            class_id = int(index.item())
            label = get_model_label(model, class_id)
            score = float(probability[class_id].item())
            print(f"  {CHINESE_LABELS.get(label, label)}：{score:.4f}")

    # --------------------------------------------------------
    # 多Clip概率平均
    # --------------------------------------------------------
    probabilities = torch.stack(clip_probabilities, dim=0).mean(dim=0)
    sorted_indices = torch.argsort(probabilities, descending=True)

    print("\n" + "=" * 60)
    print(f"{len(clips)} 个 Clip 概率平均后的最终概率")
    print("=" * 60)

    for index in sorted_indices:
        class_id = int(index.item())
        label = get_model_label(model, class_id)
        score = float(probabilities[class_id].item())
        print(f"{CHINESE_LABELS.get(label, label)}：{score:.4f}")

    best_id = int(sorted_indices[0].item())
    second_id = int(sorted_indices[1].item())

    best_label = get_model_label(model, best_id)
    second_label = get_model_label(model, second_id)

    best_score = float(probabilities[best_id].item())
    second_score = float(probabilities[second_id].item())
    margin = best_score - second_score

    print("\n" + "-" * 60)
    print("识别结果")
    print("-" * 60)
    print(f"最高类别：{CHINESE_LABELS.get(best_label, best_label)}")
    print(f"最高置信度：{best_score:.4f}")
    print(f"第二类别：{CHINESE_LABELS.get(second_label, second_label)}")
    print(f"第二高置信度：{second_score:.4f}")
    print(f"Top1-Top2 Margin：{margin:.4f}")

    # --------------------------------------------------------
    # 拒识/不确定规则
    # --------------------------------------------------------
    if best_score < UNKNOWN_THRESHOLD:
        print("\n最终结果：无法识别投篮方式")
        print("原因：三种已知类别的最高概率仍然过低。")

    elif best_score < CONFIDENCE_THRESHOLD:
        print("\n最终结果：不确定")
        print(f"可能类别：{CHINESE_LABELS.get(best_label, best_label)}")
        print("原因：最高类别置信度不足。")

    elif margin < MARGIN_THRESHOLD:
        print("\n最终结果：不确定")
        print("模型主要在以下两个类别之间犹豫：")
        print(f"1. {CHINESE_LABELS.get(best_label, best_label)}：{best_score:.4f}")
        print(f"2. {CHINESE_LABELS.get(second_label, second_label)}：{second_score:.4f}")
        print("原因：第一名与第二名概率差距较小。")

    else:
        print(
            "\n最终结果："
            f"{CHINESE_LABELS.get(best_label, best_label)}"
        )
        print(f"置信度：{best_score:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
