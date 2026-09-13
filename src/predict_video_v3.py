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
MODEL_DIR = PROJECT_ROOT / "outputs" / "videomae_v3" / "best_model"
VIDEO_PATH = PROJECT_ROOT / "demo" / "2.mp4"


# ============================================================
# 2. 类别
# ============================================================
CHINESE_LABELS = {
    "one_stage": "一段式",
    "one_point_five_stage": "一点五段式",
    "two_stage": "二段式",
}


# ============================================================
# 3. V3推理配置
# ============================================================
NUM_CLIPS = 5
TEMPORAL_WINDOW_RATIO = 0.80

# 这三个阈值不会提高closed-set accuracy，仅用于是否输出“不确定”。
# Label smoothing已经改为0.05，后续最好根据VAL重新标定。
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
# 5. V3 Temporal Multi-View
# ============================================================
def uniform_indices(start: int, end_exclusive: int, num_frames: int) -> np.ndarray:
    if end_exclusive <= start:
        end_exclusive = start + 1
    return np.linspace(
        start,
        end_exclusive - 1,
        num=num_frames,
    ).round().astype(int)


def temporal_multiview_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    num_clips: int = NUM_CLIPS,
    window_ratio: float = TEMPORAL_WINDOW_RATIO,
) -> list[list[np.ndarray]]:
    """
    与V3训练/验证一致：
    每个clip是一个连续时间窗口，而不是在整个视频16段中取固定相对位置。
    """
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total <= num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        return [[frames[i] for i in indices]]

    window_len = max(num_frames, min(int(round(total * window_ratio)), total))
    max_start = total - window_len

    if num_clips <= 1 or max_start <= 0:
        starts = [max_start // 2]
    else:
        starts = np.linspace(0, max_start, num=num_clips).round().astype(int).tolist()

    clips: list[list[np.ndarray]] = []
    for start in starts:
        end = start + window_len
        indices = uniform_indices(start, end, num_frames)
        clips.append([frames[i] for i in indices])

    return clips


def get_model_label(
    model: VideoMAEForVideoClassification,
    class_id: int,
) -> str:
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
    print("篮球投篮方式识别 - VideoMAE V3")
    print("=" * 60)
    print(f"设备：{device}")
    print(f"模型：{MODEL_DIR}")
    print(f"视频：{VIDEO_PATH}")
    print(f"Temporal Clips：{NUM_CLIPS}")
    print(f"Window Ratio：{TEMPORAL_WINDOW_RATIO:.0%}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    frames = read_all_frames(VIDEO_PATH)
    clips = temporal_multiview_sample_frames(
        frames,
        num_frames=int(model.config.num_frames),
        num_clips=NUM_CLIPS,
        window_ratio=TEMPORAL_WINDOW_RATIO,
    )

    print(f"视频总帧数：{len(frames)}")
    print(f"实际生成Clip数量：{len(clips)}")

    clip_logits: list[torch.Tensor] = []

    with torch.inference_mode():
        for clip in clips:
            inputs = processor(clip, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values).logits[0]
            clip_logits.append(logits.detach().float().cpu())

    # --------------------------------------------------------
    # 每个Clip的结果
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("各Temporal Clip预测")
    print("=" * 60)

    for clip_index, logits in enumerate(clip_logits, start=1):
        probability = torch.softmax(logits, dim=-1)
        order = torch.argsort(probability, descending=True)
        print(f"\nClip {clip_index}")
        for index in order:
            class_id = int(index.item())
            label = get_model_label(model, class_id)
            score = float(probability[class_id].item())
            print(f"  {CHINESE_LABELS.get(label, label)}：{score:.4f}")

    # --------------------------------------------------------
    # V3：Logits Mean -> Softmax
    # --------------------------------------------------------
    avg_logits = torch.stack(clip_logits, dim=0).mean(dim=0)
    probabilities = torch.softmax(avg_logits, dim=-1)
    sorted_indices = torch.argsort(probabilities, descending=True)

    print("\n" + "=" * 60)
    print(f"{len(clips)}个Clip Logits平均后的最终概率")
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
        print(f"\n最终结果：{CHINESE_LABELS.get(best_label, best_label)}")
        print(f"置信度：{best_score:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
