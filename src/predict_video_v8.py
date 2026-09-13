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
MODEL_DIR = PROJECT_ROOT / "outputs" / "videomae_v8" / "best_model"

# 修改为需要预测的视频
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
# 3. V8推理配置
# ============================================================
EXPECTED_NUM_FRAMES = 32
NUM_CLIPS = 3
TEMPORAL_WINDOW_RATIO = 0.80
CENTER_CLIP_SHIFT_RATIO = 0.05
CENTER_CLIP_WEIGHTS = (0.25, 0.50, 0.25)
CLIP_NAMES = ("中心前偏", "正中心", "中心后偏")

# 这三个阈值只用于拒识/不确定输出，不会提高closed-set三分类Accuracy。
# V8保持Hard Cross Entropy；这些阈值仍建议基于固定VAL集重新标定。
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
# 5. V8中心附近3-Clip时间采样
# ============================================================
def uniform_indices(
    start: int,
    end_exclusive: int,
    num_frames: int,
) -> np.ndarray:
    if end_exclusive <= start:
        end_exclusive = start + 1
    return np.linspace(
        start,
        end_exclusive - 1,
        num=num_frames,
    ).round().astype(int)


def centered_weighted_multiclip_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    window_ratio: float = TEMPORAL_WINDOW_RATIO,
    shift_ratio: float = CENTER_CLIP_SHIFT_RATIO,
) -> tuple[list[list[np.ndarray]], list[int]]:
    """
    生成3个都围绕视频中心的连续窗口：
      1. 中心向前轻微偏移
      2. 正中心
      3. 中心向后轻微偏移

    每个窗口内部均匀采样VideoMAE要求的num_frames帧。
    """
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total <= num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        clip = [frames[i] for i in indices]
        return [clip, clip, clip], [0, 0, 0]

    window_len = max(num_frames, min(int(round(total * window_ratio)), total))
    max_start = max(0, total - window_len)
    center_start = max_start // 2
    shift = int(round(total * shift_ratio))

    starts = [
        max(0, min(center_start - shift, max_start)),
        max(0, min(center_start, max_start)),
        max(0, min(center_start + shift, max_start)),
    ]

    clips: list[list[np.ndarray]] = []
    for clip_start in starts:
        clip_end = clip_start + window_len
        indices = uniform_indices(clip_start, clip_end, num_frames)
        clips.append([frames[i] for i in indices])

    return clips, starts


def weighted_mean_logits(
    clip_logits: list[torch.Tensor],
    weights: tuple[float, ...] = CENTER_CLIP_WEIGHTS,
) -> torch.Tensor:
    if len(clip_logits) != len(weights):
        raise ValueError(
            f"Clip数量({len(clip_logits)})与权重数量({len(weights)})不一致"
        )

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    weight_tensor = weight_tensor / weight_tensor.sum()
    stacked = torch.stack([x.float().cpu() for x in clip_logits], dim=0)
    return (stacked * weight_tensor[:, None]).sum(dim=0)


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
    print("篮球投篮方式识别 - VideoMAE V8")
    print("=" * 60)
    print(f"设备：{device}")
    print(f"模型：{MODEL_DIR}")
    print(f"视频：{VIDEO_PATH}")
    print(f"中心Clip数量：{NUM_CLIPS}")
    print(f"Window Ratio：{TEMPORAL_WINDOW_RATIO:.0%}")
    print(f"中心偏移比例：±{CENTER_CLIP_SHIFT_RATIO:.0%}")
    print(f"融合权重：{CENTER_CLIP_WEIGHTS}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    actual_num_frames = int(model.config.num_frames)
    if actual_num_frames != EXPECTED_NUM_FRAMES:
        raise RuntimeError(
            f"V8模型帧数异常：期望{EXPECTED_NUM_FRAMES}，实际{actual_num_frames}。"
            "请确认加载的是outputs/videomae_v8/best_model。"
        )

    tubelet_size = int(model.config.tubelet_size)
    pos_shape = tuple(model.videomae.embeddings.position_embeddings.shape)
    print(f"模型输入帧数：{actual_num_frames}")
    print(f"Tubelet Size：{tubelet_size}")
    print(f"固定sin-cos Position Shape：{pos_shape}")

    frames = read_all_frames(VIDEO_PATH)
    clips, starts = centered_weighted_multiclip_sample_frames(
        frames,
        num_frames=actual_num_frames,
        window_ratio=TEMPORAL_WINDOW_RATIO,
        shift_ratio=CENTER_CLIP_SHIFT_RATIO,
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
    # 每个Clip单独结果
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("各中心Temporal Clip预测")
    print("=" * 60)

    for clip_index, (name, logits, start, weight) in enumerate(
        zip(CLIP_NAMES, clip_logits, starts, CENTER_CLIP_WEIGHTS),
        start=1,
    ):
        probability = torch.softmax(logits, dim=-1)
        order = torch.argsort(probability, descending=True)
        print(
            f"\nClip {clip_index} [{name}] "
            f"start_frame={start}, fusion_weight={weight:.2f}"
        )
        for index in order:
            class_id = int(index.item())
            label = get_model_label(model, class_id)
            score = float(probability[class_id].item())
            print(f"  {CHINESE_LABELS.get(label, label)}：{score:.4f}")

    # --------------------------------------------------------
    # V8：中心3-Clip加权Logits -> Softmax
    # --------------------------------------------------------
    avg_logits = weighted_mean_logits(
        clip_logits,
        weights=CENTER_CLIP_WEIGHTS,
    )
    probabilities = torch.softmax(avg_logits, dim=-1)
    sorted_indices = torch.argsort(probabilities, descending=True)

    print("\n" + "=" * 60)
    print("中心3-Clip加权Logits融合后的最终概率")
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
