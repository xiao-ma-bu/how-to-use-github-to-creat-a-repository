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
MODEL_DIR = PROJECT_ROOT / "outputs" / "videomae_v6" / "best_model"

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
# 3. V6固定物理时间推理配置
# ============================================================
NUM_CLIPS = 3
TARGET_FPS = 30.0
FIXED_WINDOW_SECONDS = 1.50
CENTER_CLIP_SHIFT_SECONDS = 0.10
CENTER_CLIP_WEIGHTS = (0.25, 0.50, 0.25)
CLIP_NAMES = ("中心前偏", "正中心", "中心后偏")

# 只用于拒识/不确定输出，不改变closed-set三分类Accuracy。
# 建议以后只在固定VAL集上标定，不使用TEST调阈值。
UNKNOWN_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.30


# ============================================================
# 4. 视频读取
# ============================================================
def read_all_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
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
    if fps <= 0:
        raise RuntimeError(f"视频FPS异常：{fps}")

    return frames, fps


# ============================================================
# 5. V6：固定FPS + 固定1.5秒时间窗口
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


def resample_frames_to_target_fps(
    frames: list[np.ndarray],
    source_fps: float,
    target_fps: float = TARGET_FPS,
) -> list[np.ndarray]:
    """
    把源视频映射到统一target_fps时间轴，但不改变动作真实时长。
    只做基于时间戳的取帧/邻近帧重复，不做光流插帧。
    """
    if not frames:
        raise RuntimeError("视频没有有效帧")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"FPS必须为正数：source={source_fps}, target={target_fps}")

    if len(frames) == 1:
        return [frames[0]]

    duration_seconds = (len(frames) - 1) / source_fps
    target_count = max(2, int(round(duration_seconds * target_fps)) + 1)
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    source_indices = np.rint(target_times * source_fps).astype(int)
    source_indices = np.clip(source_indices, 0, len(frames) - 1)
    return [frames[int(i)] for i in source_indices]


def fixed_window_frame_count(
    window_seconds: float = FIXED_WINDOW_SECONDS,
    target_fps: float = TARGET_FPS,
) -> int:
    return max(2, int(round(window_seconds * target_fps)) + 1)


def pad_frames_to_min_length(
    frames: list[np.ndarray],
    min_length: int,
) -> tuple[list[np.ndarray], int, int]:
    """短视频首尾复制边界帧，不拉伸动作时间轴。"""
    if not frames:
        raise RuntimeError("视频没有有效帧")
    if len(frames) >= min_length:
        return frames, 0, 0

    missing = min_length - len(frames)
    left_pad = missing // 2
    right_pad = missing - left_pad
    padded = [frames[0]] * left_pad + frames + [frames[-1]] * right_pad
    return padded, left_pad, right_pad


def sample_from_fixed_window(
    normalized_frames: list[np.ndarray],
    num_frames: int,
    start: int,
    window_frames: int,
) -> list[np.ndarray]:
    indices = uniform_indices(start, start + window_frames, num_frames)
    return [normalized_frames[int(i)] for i in indices]


def centered_weighted_multiclip_sample_frames(
    frames: list[np.ndarray],
    source_fps: float,
    num_frames: int,
    shift_seconds: float = CENTER_CLIP_SHIFT_SECONDS,
) -> tuple[list[list[np.ndarray]], list[int], dict[str, float]]:
    """
    三个固定1.5秒窗口：中心前偏、正中心、中心后偏。
    每个窗口内部均匀抽取模型要求的num_frames帧。
    """
    normalized = resample_frames_to_target_fps(
        frames,
        source_fps=source_fps,
        target_fps=TARGET_FPS,
    )

    normalized_before_padding = len(normalized)
    window_frames = fixed_window_frame_count()
    normalized, left_pad, right_pad = pad_frames_to_min_length(
        normalized,
        window_frames,
    )

    max_start = len(normalized) - window_frames
    center_start = max_start // 2
    shift_frames = int(round(shift_seconds * TARGET_FPS))

    starts = [
        max(0, min(center_start - shift_frames, max_start)),
        max(0, min(center_start, max_start)),
        max(0, min(center_start + shift_frames, max_start)),
    ]

    clips = [
        sample_from_fixed_window(
            normalized,
            num_frames=num_frames,
            start=start,
            window_frames=window_frames,
        )
        for start in starts
    ]

    metadata = {
        "source_fps": float(source_fps),
        "source_frame_count": float(len(frames)),
        "source_duration_seconds": float((len(frames) - 1) / source_fps),
        "target_fps": float(TARGET_FPS),
        "normalized_frame_count_before_padding": float(normalized_before_padding),
        "left_pad_frames": float(left_pad),
        "right_pad_frames": float(right_pad),
        "fixed_window_seconds": float(FIXED_WINDOW_SECONDS),
        "fixed_window_frames": float(window_frames),
        "shift_seconds": float(shift_seconds),
    }
    return clips, starts, metadata


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

    print("=" * 70)
    print("篮球投篮方式识别 - VideoMAE V6")
    print("固定FPS + 固定物理时间窗口 + 中心加权3-Clip")
    print("=" * 70)
    print(f"设备：{device}")
    print(f"模型：{MODEL_DIR}")
    print(f"视频：{VIDEO_PATH}")
    print(f"Target FPS：{TARGET_FPS}")
    print(f"固定时间窗口：{FIXED_WINDOW_SECONDS:.2f} 秒")
    print(f"中心偏移：±{CENTER_CLIP_SHIFT_SECONDS:.2f} 秒")
    print(f"融合权重：{CENTER_CLIP_WEIGHTS}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    frames, source_fps = read_all_frames(VIDEO_PATH)
    clips, starts, metadata = centered_weighted_multiclip_sample_frames(
        frames,
        source_fps=source_fps,
        num_frames=int(model.config.num_frames),
        shift_seconds=CENTER_CLIP_SHIFT_SECONDS,
    )

    print("\n" + "-" * 70)
    print("视频时间信息")
    print("-" * 70)
    print(f"源FPS：{metadata['source_fps']:.3f}")
    print(f"源总帧数：{int(metadata['source_frame_count'])}")
    print(f"源时长：{metadata['source_duration_seconds']:.3f} 秒")
    print(f"统一时间轴FPS：{metadata['target_fps']:.1f}")
    print(
        "统一时间轴帧数："
        f"{int(metadata['normalized_frame_count_before_padding'])}"
    )
    if metadata["left_pad_frames"] > 0 or metadata["right_pad_frames"] > 0:
        print(
            "短视频边界补帧："
            f"left={int(metadata['left_pad_frames'])}, "
            f"right={int(metadata['right_pad_frames'])}"
        )

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
    print("\n" + "=" * 70)
    print("各Fixed-Time Clip预测")
    print("=" * 70)

    for clip_index, (name, logits, start, weight) in enumerate(
        zip(CLIP_NAMES, clip_logits, starts, CENTER_CLIP_WEIGHTS),
        start=1,
    ):
        probability = torch.softmax(logits, dim=-1)
        order = torch.argsort(probability, descending=True)
        start_seconds = start / TARGET_FPS
        end_seconds = start_seconds + FIXED_WINDOW_SECONDS

        print(
            f"\nClip {clip_index} [{name}] "
            f"normalized_start={start_seconds:.3f}s, "
            f"normalized_end={end_seconds:.3f}s, "
            f"fusion_weight={weight:.2f}"
        )

        for index in order:
            class_id = int(index.item())
            label = get_model_label(model, class_id)
            score = float(probability[class_id].item())
            print(f"  {CHINESE_LABELS.get(label, label)}：{score:.4f}")

    # --------------------------------------------------------
    # V6：固定时间3-Clip加权Logits -> Softmax
    # --------------------------------------------------------
    avg_logits = weighted_mean_logits(
        clip_logits,
        weights=CENTER_CLIP_WEIGHTS,
    )
    probabilities = torch.softmax(avg_logits, dim=-1)
    sorted_indices = torch.argsort(probabilities, descending=True)

    print("\n" + "=" * 70)
    print("固定时间3-Clip加权Logits融合后的最终概率")
    print("=" * 70)

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

    print("\n" + "-" * 70)
    print("识别结果")
    print("-" * 70)
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

    print("=" * 70)


if __name__ == "__main__":
    main()
