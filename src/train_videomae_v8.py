from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    VideoMAEConfig,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    set_seed,
)


# ============================================================
# 1. 路径与类别
# ============================================================
PROJECT_ROOT = Path(r"F:\BasketballVideoMAE")
DATA_ROOT = PROJECT_ROOT / "data" / "split"
MODEL_NAME = PROJECT_ROOT / "models" / "videomae-base"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videomae_v8"

CLASS_NAMES = [
    "one_stage",
    "one_point_five_stage",
    "two_stage",
]

CHINESE_LABELS = {
    "one_stage": "一段式",
    "one_point_five_stage": "一点五段式",
    "two_stage": "二段式",
}

LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ============================================================
# 2. V8训练参数
#    严格以V5为Baseline，只改变一个核心变量：输入帧数16 -> 32。
#    ① Stage2仍然全网络解冻 + V5分层学习率
#    ② 仍使用90~100%连续训练时间窗口
#    ③ 仍使用中心附近3-Clip + 0.25/0.50/0.25加权Logits融合
#    ④ 仍使用sqrt Player-Class软平衡，不做类别总质量1:1归一化
#    ⑤ 仍使用Hard Cross Entropy，无Label Smoothing/Ordinal Soft Target
#    ⑥ 仍使用max_grad_norm=1.0与相同Early Stopping
#    ⑦ 唯一训练输入变化：VideoMAE num_frames = 32
#    ⑧ HF VideoMAE位置编码是固定sin-cos；按32帧config重新生成对应长度，
#       不对不存在的可学习position parameter做插值。
# ============================================================
RANDOM_SEED = 42

# V8严格消融：V5其余训练策略保持不变，只把模型输入从16帧提高到32帧。
INPUT_NUM_FRAMES = 32

# 固定V5/V6/V7使用的同一Train/Val/Test划分，保证V8可直接比较。
EXPECTED_SPLIT_SIGNATURE = (
    "73cce1aaaa40004fad9746b211bd154410015cc3c48fddffb503913869c34bff"
)

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.10

# V8保持V5的Hard Cross Entropy，不做Label Smoothing/Ordinal Soft Target
LABEL_SMOOTHING_FACTOR = 0.0
ENABLE_ORDINAL_SOFT_TARGET = False

# Stage 1：只训练分类头 + fc_norm
STAGE1_EPOCHS = 4
STAGE1_LEARNING_RATE = 1e-4

# Stage 2：全网络解冻 + 分层学习率
STAGE2_EPOCHS = 25
CLASSIFIER_LR = 5e-5
TOP_ENCODER_LR = 2e-5
MID_ENCODER_LR = 1e-5
BOTTOM_ENCODER_LR = 5e-6
EMBEDDING_LR = 2e-6

ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_THRESHOLD = 0.0

# 训练时尽量保留完整动作链
TRAIN_TEMPORAL_MIN_RATIO = 0.90
TRAIN_TEMPORAL_MAX_RATIO = 1.00

# Single-Clip中心窗口
VAL_TEMPORAL_WINDOW_RATIO = 0.80
TEST_TEMPORAL_WINDOW_RATIO = 0.80

# V8：保持V5中心附近3-Clip
NUM_VAL_CLIPS = 3
NUM_TEST_CLIPS = 3
CENTER_CLIP_SHIFT_RATIO = 0.05
CENTER_CLIP_WEIGHTS = (0.25, 0.50, 0.25)

# Player-Class软平衡：只抑制高频player-class组，默认不强制三类总采样质量1:1
ENABLE_PLAYER_BALANCE = True
PLAYER_BALANCE_POWER = 0.50
ENABLE_CLASS_BALANCE = False

# 梯度稳定
MAX_GRAD_NORM = 1.0

# Stage2验证诊断
SAVE_VAL_EPOCH_DIAGNOSTICS = True
COLLAPSE_WARNING_THRESHOLD = 0.60

# 训练增强：继续保持视频级一致性增强
ENABLE_VIDEO_AUGMENTATION = True
HORIZONTAL_FLIP_PROB = 0.50
RANDOM_CROP_MIN_SCALE = 0.85
RANDOM_CROP_MAX_SCALE = 1.00
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.85, 1.15)
SATURATION_RANGE = (0.90, 1.10)
GAUSSIAN_BLUR_PROB = 0.10

# 视频检查
PRECHECK_ALL_VIDEOS = True
PRECHECK_PRINT_EVERY = 10
MAX_VIDEO_READ_RETRIES = 3
VIDEO_RETRY_WAIT_SECONDS = 0.5
MAX_ALLOWED_FRAME_DIFFERENCE = 3
FRAME_COUNT_TOLERANCE_RATIO = 0.02

# 每次正式实验清理V8旧stage输出，避免误恢复旧checkpoint
CLEAN_OLD_STAGE_OUTPUTS = True


# ============================================================
# 3. 随机种子
# ============================================================
set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 4. 视频读取与完整性检查
# ============================================================
def get_allowed_frame_difference(expected_frame_count: int) -> int:
    if expected_frame_count <= 0:
        return MAX_ALLOWED_FRAME_DIFFERENCE
    ratio_difference = int(np.ceil(expected_frame_count * FRAME_COUNT_TOLERANCE_RATIO))
    return max(MAX_ALLOWED_FRAME_DIFFERENCE, ratio_difference)


def check_video_decodable(video_path: Path) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False, "无法打开视频"

    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        decoded += 1
    cap.release()

    if decoded == 0:
        return False, "没有成功解码任何帧"
    if decoded < 8:
        return False, f"实际只能解码 {decoded} 帧，帧数过少"
    if fps <= 0:
        return False, f"FPS异常：{fps}"

    if expected > 0:
        difference = expected - decoded
        if difference > get_allowed_frame_difference(expected):
            return (
                False,
                f"视频未完整解码：声明 {expected} 帧，实际 {decoded} 帧，"
                f"解码比例 {decoded / expected:.2%}",
            )

    return True, f"{decoded}帧，{fps:.2f} FPS，{width}×{height}"


def read_all_frames(
    video_path: Path,
    max_retries: int = MAX_VIDEO_READ_RETRIES,
) -> list[np.ndarray]:
    last_error = "未知错误"

    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            last_error = "无法打开视频"
            cap.release()
            time.sleep(VIDEO_RETRY_WAIT_SECONDS)
            continue

        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[np.ndarray] = []

        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        decoded = len(frames)
        if decoded == 0:
            last_error = "没有读取到有效帧"
            time.sleep(VIDEO_RETRY_WAIT_SECONDS)
            continue
        if decoded < 8:
            last_error = f"只能解码 {decoded} 帧"
            time.sleep(VIDEO_RETRY_WAIT_SECONDS)
            continue

        if expected > 0:
            difference = expected - decoded
            if difference > get_allowed_frame_difference(expected):
                last_error = (
                    f"视频未完整解码：声明 {expected} 帧，实际 {decoded} 帧，"
                    f"比例 {decoded / expected:.2%}"
                )
                time.sleep(VIDEO_RETRY_WAIT_SECONDS)
                continue

        if attempt > 1:
            print(f"[视频读取恢复] 第 {attempt} 次成功：{video_path}")
        return frames

    raise RuntimeError(
        f"视频连续多次读取失败：{video_path}\n"
        f"重试次数：{max_retries}\n最后错误：{last_error}"
    )


# ============================================================
# 5. V4时间采样
# ============================================================
def uniform_indices(start: int, end_exclusive: int, num_frames: int) -> np.ndarray:
    """在[start, end_exclusive)连续窗口内均匀采样num_frames帧。"""
    if end_exclusive <= start:
        end_exclusive = start + 1
    indices = np.linspace(
        start,
        end_exclusive - 1,
        num=num_frames,
    ).round().astype(int)
    return indices


def random_temporal_window_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    min_ratio: float = TRAIN_TEMPORAL_MIN_RATIO,
    max_ratio: float = TRAIN_TEMPORAL_MAX_RATIO,
) -> list[np.ndarray]:
    """
    TRAIN：随机选择一个连续时间窗口，再在窗口中均匀抽取num_frames帧。
    相比“每段随机一帧”，更保留真实帧间时序结构。
    """
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total <= num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        return [frames[i] for i in indices]

    ratio = random.uniform(min_ratio, max_ratio)
    window_len = int(round(total * ratio))
    window_len = max(num_frames, min(window_len, total))

    max_start = total - window_len
    start = random.randint(0, max_start) if max_start > 0 else 0
    end = start + window_len

    indices = uniform_indices(start, end, num_frames)
    return [frames[i] for i in indices]


def center_temporal_window_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    window_ratio: float = VAL_TEMPORAL_WINDOW_RATIO,
) -> list[np.ndarray]:
    """Single-Clip验证/测试：取居中的连续窗口并均匀采样。"""
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total <= num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        return [frames[i] for i in indices]

    window_len = max(num_frames, min(int(round(total * window_ratio)), total))
    start = max(0, (total - window_len) // 2)
    end = start + window_len
    indices = uniform_indices(start, end, num_frames)
    return [frames[i] for i in indices]


def centered_weighted_multiclip_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    window_ratio: float,
    shift_ratio: float = CENTER_CLIP_SHIFT_RATIO,
) -> list[list[np.ndarray]]:
    """
    V8中心附近3-Clip：
      Clip 1：中心窗口向前轻微偏移
      Clip 2：正中心窗口
      Clip 3：中心窗口向后轻微偏移

    保留V4验证有效的中心附近设计，尽量保证三个Clip都包含
    set point -> 伸肘 -> 出手这一关键动作区域。
    """
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total <= num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        clip = [frames[i] for i in indices]
        return [clip, clip, clip]

    window_len = max(num_frames, min(int(round(total * window_ratio)), total))
    max_start = max(0, total - window_len)
    center_start = max_start // 2

    # shift按完整视频长度定义，并限制在合法范围内
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

    return clips


def weighted_mean_logits(
    clip_logits: list[torch.Tensor],
    weights: tuple[float, ...] = CENTER_CLIP_WEIGHTS,
) -> torch.Tensor:
    """对多个Clip的Logits进行归一化加权平均。"""
    if not clip_logits:
        raise RuntimeError("clip_logits为空")

    if len(clip_logits) != len(weights):
        raise ValueError(
            f"Clip数量({len(clip_logits)})与权重数量({len(weights)})不一致"
        )

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    weight_tensor = weight_tensor / weight_tensor.sum()
    stacked = torch.stack([x.float().cpu() for x in clip_logits], dim=0)
    return (stacked * weight_tensor[:, None]).sum(dim=0)


# ============================================================
# 6. 视频一致性空间增强
# ============================================================
def apply_consistent_video_augmentation(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames or not ENABLE_VIDEO_AUGMENTATION:
        return frames

    height, width = frames[0].shape[:2]

    # 同一clip使用相同的crop参数
    scale = random.uniform(RANDOM_CROP_MIN_SCALE, RANDOM_CROP_MAX_SCALE)
    crop_h = max(1, int(round(height * math.sqrt(scale))))
    crop_w = max(1, int(round(width * math.sqrt(scale))))
    crop_h = min(crop_h, height)
    crop_w = min(crop_w, width)

    max_y = height - crop_h
    max_x = width - crop_w
    top = random.randint(0, max_y) if max_y > 0 else 0
    left = random.randint(0, max_x) if max_x > 0 else 0

    do_flip = random.random() < HORIZONTAL_FLIP_PROB
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    contrast = random.uniform(*CONTRAST_RANGE)
    saturation = random.uniform(*SATURATION_RANGE)
    do_blur = random.random() < GAUSSIAN_BLUR_PROB

    output: list[np.ndarray] = []
    for frame in frames:
        img = frame[top: top + crop_h, left: left + crop_w]
        if crop_h != height or crop_w != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        if do_flip:
            img = np.ascontiguousarray(img[:, ::-1])

        # brightness + contrast，所有帧参数一致
        img_float = img.astype(np.float32)
        mean = img_float.mean(axis=(0, 1), keepdims=True)
        img_float = (img_float - mean) * contrast + mean
        img_float = img_float * brightness
        img_float = np.clip(img_float, 0, 255).astype(np.uint8)

        # saturation，所有帧参数一致
        hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if do_blur:
            img_aug = cv2.GaussianBlur(img_aug, (3, 3), 0)

        output.append(np.ascontiguousarray(img_aug))

    return output


# ============================================================
# 7. Dataset
# ============================================================
def parse_player_id(video_path: Path) -> str:
    player_id = video_path.stem.split("_")[0].strip()
    if not player_id:
        raise ValueError(f"无法解析球员ID：{video_path.name}")
    return player_id


class BasketballVideoDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        processor: VideoMAEImageProcessor,
        num_frames: int,
        training: bool,
    ) -> None:
        self.root_dir = root_dir
        self.processor = processor
        self.num_frames = num_frames
        self.training = training
        self.samples: list[tuple[Path, int]] = []
        self.player_ids: list[str] = []

        for class_name in CLASS_NAMES:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"缺少类别文件夹：{class_dir}")

            label_id = LABEL_TO_ID[class_name]
            for video_path in sorted(class_dir.rglob("*")):
                if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    self.samples.append((video_path, label_id))
                    self.player_ids.append(parse_player_id(video_path))

        if not self.samples:
            raise RuntimeError(f"数据集中没有视频：{root_dir}")

        print(f"{root_dir.name}：{len(self.samples)} 个视频")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        video_path, label = self.samples[index]
        frames = read_all_frames(video_path)

        if self.training:
            frames = random_temporal_window_sample_frames(
                frames,
                self.num_frames,
                TRAIN_TEMPORAL_MIN_RATIO,
                TRAIN_TEMPORAL_MAX_RATIO,
            )
            frames = apply_consistent_video_augmentation(frames)
        else:
            frames = center_temporal_window_sample_frames(
                frames,
                self.num_frames,
                VAL_TEMPORAL_WINDOW_RATIO,
            )

        processed = self.processor(frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
            "video_path": str(video_path),
        }


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in examples]),
        "labels": torch.stack([x["labels"] for x in examples]),
    }


# ============================================================
# 8. V8：Player-Class软平衡 + Hard Cross Entropy
# ============================================================
def build_player_class_sample_weights(dataset: BasketballVideoDataset) -> torch.DoubleTensor:
    """
    V8沿用V5的默认策略：

    player-class组内每条样本：
        raw_weight = group_size ^ (-PLAYER_BALANCE_POWER)

    PLAYER_BALANCE_POWER=0.5时：
        1条视频组的单样本权重 = 1
        25条视频组的单样本权重 = 1/5

    与V4不同：ENABLE_CLASS_BALANCE=False时，不再把三个类别的总采样质量
    强制归一到1:1:1，避免少样本类别被过度重复采样。
    """
    if not ENABLE_PLAYER_BALANCE:
        return torch.ones(len(dataset), dtype=torch.double)

    group_counts: Counter[tuple[str, int]] = Counter()
    for player_id, (_, label) in zip(dataset.player_ids, dataset.samples):
        group_counts[(player_id, label)] += 1

    weights: list[float] = []
    for player_id, (_, label) in zip(dataset.player_ids, dataset.samples):
        group_size = group_counts[(player_id, label)]
        raw = 1.0 / (float(group_size) ** PLAYER_BALANCE_POWER)
        weights.append(raw)

    tensor = torch.tensor(weights, dtype=torch.double)

    if ENABLE_CLASS_BALANCE:
        class_mass: defaultdict[int, float] = defaultdict(float)
        for weight, (_, label) in zip(tensor.tolist(), dataset.samples):
            class_mass[label] += float(weight)

        corrected: list[float] = []
        for weight, (_, label) in zip(tensor.tolist(), dataset.samples):
            corrected.append(float(weight) / max(class_mass[label], 1e-12))
        tensor = torch.tensor(corrected, dtype=torch.double)

    tensor = tensor / tensor.mean()

    print("\n" + "=" * 70)
    print("V8 Player-Class Soft Sampling")
    print("=" * 70)
    print(f"Player balance：{ENABLE_PLAYER_BALANCE}")
    print(f"Player balance power：{PLAYER_BALANCE_POWER}")
    print(f"Class mass balance：{ENABLE_CLASS_BALANCE}")
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == class_id]
        players = {dataset.player_ids[i] for i in class_indices}
        class_weight_sum = float(tensor[class_indices].sum().item()) if class_indices else 0.0
        print(
            f"{class_name:<24} videos={len(class_indices):<4} "
            f"players={len(players):<3} normalized_weight_sum={class_weight_sum:.4f}"
        )
    print(f"样本权重范围：{tensor.min().item():.4f} ~ {tensor.max().item():.4f}")
    print("=" * 70)
    return tensor


class PlayerBalancedTrainer(Trainer):
    """V8：Hard Cross Entropy由Transformers默认模型loss负责，仅覆盖训练Sampler。"""

    def _get_train_sampler(self, train_dataset: Dataset | None = None):
        if not ENABLE_PLAYER_BALANCE:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not isinstance(dataset, BasketballVideoDataset):
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

        weights = build_player_class_sample_weights(dataset)
        generator = torch.Generator()
        generator.manual_seed(RANDOM_SEED)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )


# ============================================================
# 9. 指标
# ============================================================
def metrics_from_ids(
    labels: np.ndarray,
    predictions: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    accuracy = accuracy_score(labels, predictions)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    class_p, class_r, class_f1, class_support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )

    out: dict[str, float] = {
        f"{prefix}accuracy": float(accuracy),
        f"{prefix}macro_precision": float(macro_p),
        f"{prefix}macro_recall": float(macro_r),
        f"{prefix}macro_f1": float(macro_f1),
    }

    for i, class_name in enumerate(CLASS_NAMES):
        out[f"{prefix}{class_name}_precision"] = float(class_p[i])
        out[f"{prefix}{class_name}_recall"] = float(class_r[i])
        out[f"{prefix}{class_name}_f1"] = float(class_f1[i])
        out[f"{prefix}{class_name}_support"] = float(class_support[i])

    return out


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=1)
    labels = np.asarray(eval_prediction.label_ids)
    return metrics_from_ids(labels, predictions)


# ============================================================
# 10. Stage 1 / Stage 2参数控制
# ============================================================
def freeze_for_stage1(model: VideoMAEForVideoClassification) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if hasattr(model, "fc_norm") and model.fc_norm is not None:
        for parameter in model.fc_norm.parameters():
            parameter.requires_grad = True

    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    print_trainable_parameters(model, "Stage 1")


def unfreeze_for_stage2(model: VideoMAEForVideoClassification) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

    model.gradient_checkpointing_enable()
    print_trainable_parameters(model, "Stage 2")


def print_trainable_parameters(model: VideoMAEForVideoClassification, stage_name: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 70)
    print(f"{stage_name} 参数状态")
    print("=" * 70)
    print(f"总参数量：{total:,}")
    print(f"可训练参数量：{trainable:,}")
    print(f"可训练比例：{trainable / total:.4%}")
    print("=" * 70)


def calculate_warmup_steps(dataset_size: int, epochs: int) -> tuple[int, int]:
    batches_per_epoch = math.ceil(dataset_size / TRAIN_BATCH_SIZE)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / GRADIENT_ACCUMULATION_STEPS
    )
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = max(1, int(round(total_steps * WARMUP_RATIO)))
    return warmup_steps, total_steps


def make_training_args(
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    warmup_steps: int,
    use_gradient_checkpointing: bool,
    save_total_limit: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=use_gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
    )


# ============================================================
# 11. Stage2：Layer-wise Learning Rate + V8中心加权Multi-Clip Validation
# ============================================================
def get_layerwise_lr(parameter_name: str) -> tuple[str, float]:
    # classifier/fc_norm
    if parameter_name.startswith("classifier") or ".classifier" in parameter_name:
        return "classifier", CLASSIFIER_LR
    if "fc_norm" in parameter_name:
        return "classifier", CLASSIFIER_LR

    # Transformer block
    match = re.search(r"videomae\.encoder\.layer\.(\d+)\.", parameter_name)
    if match:
        layer_id = int(match.group(1))
        if layer_id <= 3:
            return "bottom_encoder", BOTTOM_ENCODER_LR
        if layer_id <= 7:
            return "mid_encoder", MID_ENCODER_LR
        return "top_encoder", TOP_ENCODER_LR

    # embeddings/patch embeddings及其它低层backbone参数
    if parameter_name.startswith("videomae.embeddings"):
        return "embedding", EMBEDDING_LR

    # 未匹配的videomae主体默认按embedding级别谨慎微调
    if parameter_name.startswith("videomae"):
        return "embedding", EMBEDDING_LR

    return "other", TOP_ENCODER_LR


def use_weight_decay(parameter_name: str) -> bool:
    lower = parameter_name.lower()
    if lower.endswith(".bias"):
        return False
    if "layernorm" in lower or "layer_norm" in lower:
        return False
    if lower.endswith("norm.weight"):
        return False
    return True


def save_validation_epoch_diagnostics(
    dataset: BasketballVideoDataset,
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    probabilities: np.ndarray,
    epoch_value: float | None,
    metric_key_prefix: str,
) -> None:
    """保存每个Stage2验证Epoch的逐视频、逐球员和类别预测分布。"""
    if not SAVE_VAL_EPOCH_DIAGNOSTICS:
        return

    epoch_num = int(round(float(epoch_value or 0.0)))
    diag_dir = OUTPUT_DIR / "val_epoch_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, ((video_path, true_label), pred_id) in enumerate(
        zip(dataset.samples, pred_ids)
    ):
        probs = probabilities[index]
        order = np.argsort(probs)[::-1]
        best = float(probs[order[0]])
        second = float(probs[order[1]])
        records.append(
            {
                "epoch": epoch_num,
                "video_path": str(video_path),
                "video_name": video_path.name,
                "player_id": parse_player_id(video_path),
                "true_label": ID_TO_LABEL[int(true_label)],
                "predicted_label": ID_TO_LABEL[int(pred_id)],
                "correct": bool(int(true_label) == int(pred_id)),
                "confidence": best,
                "top1_top2_margin": best - second,
                **{
                    f"prob_{class_name}": float(probs[class_id])
                    for class_id, class_name in enumerate(CLASS_NAMES)
                },
            }
        )

    frame = pd.DataFrame(records)
    frame.to_csv(
        diag_dir / f"epoch_{epoch_num:03d}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 球员级摘要
    player_rows: list[dict[str, Any]] = []
    for player_id, group in frame.groupby("player_id", sort=True):
        row: dict[str, Any] = {
            "epoch": epoch_num,
            "player_id": player_id,
            "video_count": int(len(group)),
            "correct_count": int(group["correct"].sum()),
            "accuracy": float(group["correct"].mean()),
        }
        for class_name in CLASS_NAMES:
            row[f"true_{class_name}_count"] = int((group["true_label"] == class_name).sum())
            row[f"pred_{class_name}_count"] = int((group["predicted_label"] == class_name).sum())
        player_rows.append(row)

    pd.DataFrame(player_rows).to_csv(
        diag_dir / f"epoch_{epoch_num:03d}_player_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 类别预测分布与塌缩检测
    pred_counts = np.bincount(pred_ids, minlength=len(CLASS_NAMES))
    pred_ratios = pred_counts / max(len(pred_ids), 1)
    max_class_id = int(np.argmax(pred_ratios))
    max_ratio = float(pred_ratios[max_class_id])
    collapse_warning = max_ratio >= COLLAPSE_WARNING_THRESHOLD

    dist_payload: dict[str, Any] = {
        "epoch": epoch_num,
        "metric_key_prefix": metric_key_prefix,
        "sample_count": int(len(pred_ids)),
        "collapse_warning": bool(collapse_warning),
        "collapse_threshold": float(COLLAPSE_WARNING_THRESHOLD),
        "dominant_predicted_class": ID_TO_LABEL[max_class_id],
        "dominant_predicted_ratio": max_ratio,
        "predicted_distribution": {
            CLASS_NAMES[i]: {
                "count": int(pred_counts[i]),
                "ratio": float(pred_ratios[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }
    with open(
        diag_dir / f"epoch_{epoch_num:03d}_distribution.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dist_payload, f, ensure_ascii=False, indent=2)

    # 混淆矩阵
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(true_ids, pred_ids, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, values_format="d")
    plt.title(f"VAL Confusion Matrix - Epoch {epoch_num}")
    plt.tight_layout()
    plt.savefig(diag_dir / f"epoch_{epoch_num:03d}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    if collapse_warning:
        print("\n" + "!" * 70)
        print(
            "[V8 WARNING] 验证集出现类别预测集中："
            f"{ID_TO_LABEL[max_class_id]} 占 {max_ratio:.2%}，"
            f"达到/超过阈值 {COLLAPSE_WARNING_THRESHOLD:.0%}。"
        )
        print("请结合该Epoch的predictions/player_summary/confusion_matrix检查类别塌缩。")
        print("!" * 70)


class LayerwiseMultiClipTrainer(PlayerBalancedTrainer):
    def __init__(
        self,
        *args,
        multi_clip_eval_dataset: BasketballVideoDataset,
        multi_clip_processor: VideoMAEImageProcessor,
        multi_clip_num_frames: int,
        multi_clip_num_clips: int = NUM_VAL_CLIPS,
        multi_clip_window_ratio: float = VAL_TEMPORAL_WINDOW_RATIO,
        multi_clip_shift_ratio: float = CENTER_CLIP_SHIFT_RATIO,
        multi_clip_weights: tuple[float, ...] = CENTER_CLIP_WEIGHTS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_clip_eval_dataset = multi_clip_eval_dataset
        self.multi_clip_processor = multi_clip_processor
        self.multi_clip_num_frames = multi_clip_num_frames
        self.multi_clip_num_clips = multi_clip_num_clips
        self.multi_clip_window_ratio = multi_clip_window_ratio
        self.multi_clip_shift_ratio = multi_clip_shift_ratio
        self.multi_clip_weights = multi_clip_weights
        self._printed_optimizer_groups = False

        if self.multi_clip_num_clips != 3:
            raise ValueError("V8中心加权Multi-Clip固定使用3个Clip")
        if len(self.multi_clip_weights) != 3:
            raise ValueError("V8 CENTER_CLIP_WEIGHTS必须包含3个权重")

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        grouped: dict[tuple[str, float, float], list[torch.nn.Parameter]] = defaultdict(list)
        summary_numel: Counter[str] = Counter()

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            group_name, lr = get_layerwise_lr(name)
            wd = WEIGHT_DECAY if use_weight_decay(name) else 0.0
            grouped[(group_name, lr, wd)].append(parameter)
            summary_numel[f"{group_name}|lr={lr:g}|wd={wd:g}"] += parameter.numel()

        optimizer_groups = []
        for (group_name, lr, wd), parameters in grouped.items():
            optimizer_groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "weight_decay": wd,
                    "group_name": group_name,
                }
            )

        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=CLASSIFIER_LR,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if not self._printed_optimizer_groups:
            print("\n" + "=" * 70)
            print("Stage 2 Layer-wise Learning Rate")
            print("=" * 70)
            for key, numel in sorted(summary_numel.items()):
                print(f"{key:<45} params={numel:,}")
            print("=" * 70)
            self._printed_optimizer_groups = True

        return self.optimizer

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        V8训练期间每个Epoch使用中心附近3-Clip Validation：
          前偏Clip 0.25 + 中心Clip 0.50 + 后偏Clip 0.25。
        返回eval_macro_f1直接参与best checkpoint与Early Stopping。
        """
        start_time = time.time()
        model = self.model
        was_training = model.training
        model.eval()
        device = self.args.device

        true_ids: list[int] = []
        pred_ids: list[int] = []
        all_probabilities: list[np.ndarray] = []

        with torch.inference_mode():
            for video_path, true_label in self.multi_clip_eval_dataset.samples:
                frames = read_all_frames(video_path)
                clips = centered_weighted_multiclip_sample_frames(
                    frames,
                    num_frames=self.multi_clip_num_frames,
                    window_ratio=self.multi_clip_window_ratio,
                    shift_ratio=self.multi_clip_shift_ratio,
                )

                clip_logits: list[torch.Tensor] = []
                for clip in clips:
                    inputs = self.multi_clip_processor(clip, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                    logits = model(pixel_values=pixel_values).logits[0]
                    clip_logits.append(logits.detach().float().cpu())

                avg_logits = weighted_mean_logits(
                    clip_logits,
                    weights=self.multi_clip_weights,
                )
                avg_prob = torch.softmax(avg_logits, dim=-1)
                pred_id = int(torch.argmax(avg_prob).item())

                true_ids.append(int(true_label))
                pred_ids.append(pred_id)
                all_probabilities.append(avg_prob.numpy())

        labels_np = np.asarray(true_ids)
        preds_np = np.asarray(pred_ids)
        probabilities_np = np.stack(all_probabilities, axis=0)
        metrics = metrics_from_ids(labels_np, preds_np, prefix=f"{metric_key_prefix}_")
        metrics[f"{metric_key_prefix}_runtime"] = float(time.time() - start_time)
        metrics[f"{metric_key_prefix}_num_clips"] = float(self.multi_clip_num_clips)

        save_validation_epoch_diagnostics(
            dataset=self.multi_clip_eval_dataset,
            true_ids=labels_np,
            pred_ids=preds_np,
            probabilities=probabilities_np,
            epoch_value=self.state.epoch,
            metric_key_prefix=metric_key_prefix,
        )

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )

        if was_training:
            model.train()

        print(
            f"\n[V8 3-Clip Weighted VAL] epoch={self.state.epoch} "
            f"accuracy={metrics[f'{metric_key_prefix}_accuracy']:.4f} "
            f"macro_f1={metrics[f'{metric_key_prefix}_macro_f1']:.4f}"
        )

        return metrics


# ============================================================
# 12. 数据统计 / 视频预检查
# ============================================================
def print_dataset_distribution(dataset: BasketballVideoDataset, split_name: str) -> None:
    labels = [label for _, label in dataset.samples]
    players = dataset.player_ids

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} 数据分布")
    print("-" * 70)
    print(f"视频总数：{len(dataset.samples)}")
    print(f"球员总数：{len(set(players))}")

    for class_id, class_name in enumerate(CLASS_NAMES):
        indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_id]
        class_players = {players[i] for i in indices}
        print(
            f"{class_name:<24} videos={len(indices):<4} "
            f"players={len(class_players):<3} "
            f"ratio={len(indices) / len(labels):.2%}"
        )


def precheck_all_videos(datasets: dict[str, BasketballVideoDataset]) -> None:
    print("\n" + "=" * 70)
    print("训练前完整视频解码检查")
    print("=" * 70)

    failed: list[dict[str, str]] = []
    total = sum(len(ds.samples) for ds in datasets.values())
    current = 0

    for split_name, dataset in datasets.items():
        print(f"\n检查 {split_name.upper()}：{len(dataset.samples)} 个视频")
        for video_path, _ in dataset.samples:
            current += 1
            ok, info = check_video_decodable(video_path)
            if not ok:
                failed.append(
                    {
                        "split": split_name,
                        "video_path": str(video_path),
                        "reason": info,
                    }
                )
                print(f"[{current}/{total}] [FAILED] {video_path.name} -> {info}")
            elif current == 1 or current % PRECHECK_PRINT_EVERY == 0 or current == total:
                print(f"[{current}/{total}] OK：{video_path.name}")

    if failed:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "video_precheck_failures.csv"
        pd.DataFrame(failed).to_csv(path, index=False, encoding="utf-8-sig")
        raise RuntimeError(f"存在异常视频，报告已保存：{path}")

    print(f"全部 {total} 个视频均通过完整解码检查。")


# ============================================================
# 13. Split固定性记录
# ============================================================
def save_split_manifest(
    datasets: dict[str, BasketballVideoDataset],
) -> str:
    """
    保存当前split成员清单并计算SHA256指纹。
    后续V6/V7只要指纹相同，就能确认使用的是同一套Train/Val/Test成员。
    """
    rows: list[dict[str, Any]] = []
    signature_lines: list[str] = []

    for split_name in ("train", "val", "test"):
        dataset = datasets[split_name]
        for video_path, label_id in dataset.samples:
            row = {
                "split": split_name,
                "class_name": ID_TO_LABEL[int(label_id)],
                "player_id": parse_player_id(video_path),
                "video_name": video_path.name,
                "relative_path": str(video_path.relative_to(DATA_ROOT)),
            }
            rows.append(row)
            signature_lines.append(
                "|".join(
                    [
                        row["split"],
                        row["class_name"],
                        row["player_id"],
                        row["relative_path"],
                    ]
                )
            )

    rows = sorted(rows, key=lambda x: (x["split"], x["class_name"], x["video_name"]))
    signature_lines = sorted(signature_lines)
    signature_text = "\n".join(signature_lines).encode("utf-8")
    signature = hashlib.sha256(signature_text).hexdigest()

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "split_signature_sha256.txt").write_text(
        signature + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("当前Split固定性指纹")
    print("=" * 70)
    print(f"SHA256：{signature}")
    print("后续实验请确认该值保持不变。")
    print("=" * 70)
    return signature


# ============================================================
# 14. 训练日志与曲线
# ============================================================
def save_trainer_log_and_curves(trainer: Trainer, prefix: str) -> None:
    logs = trainer.state.log_history
    if not logs:
        return

    pd.DataFrame(logs).to_csv(
        OUTPUT_DIR / f"{prefix}_training_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_logs = [x for x in logs if "loss" in x and "eval_loss" not in x and "step" in x]
    if train_logs:
        plt.figure(figsize=(8, 5))
        plt.plot([x["step"] for x in train_logs], [x["loss"] for x in train_logs])
        plt.xlabel("Training Step")
        plt.ylabel("Train Loss")
        plt.title(f"{prefix} Training Loss")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{prefix}_train_loss.png", dpi=200)
        plt.close()

    f1_logs = [x for x in logs if "eval_macro_f1" in x and "epoch" in x]
    if f1_logs:
        plt.figure(figsize=(8, 5))
        plt.plot(
            [x["epoch"] for x in f1_logs],
            [x["eval_macro_f1"] for x in f1_logs],
            marker="o",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Macro-F1")
        plt.title(f"{prefix} Validation Macro-F1")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{prefix}_val_macro_f1.png", dpi=200)
        plt.close()


# ============================================================
# 15. 测试结果保存
# ============================================================
def save_confusion_matrix(
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
    suffix: str,
) -> None:
    labels = list(range(len(CLASS_NAMES)))

    matrix = confusion_matrix(true_ids, predicted_ids, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {suffix}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{suffix}.png", dpi=200)
    plt.close(fig)

    norm = confusion_matrix(true_ids, predicted_ids, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(norm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, values_format=".2f")
    plt.title(f"Normalized Confusion Matrix - {suffix}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{suffix}_normalized.png", dpi=200)
    plt.close(fig)


def save_prediction_outputs(
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
    probabilities: np.ndarray,
    samples: list[tuple[Path, int]],
    suffix: str,
) -> dict[str, float]:
    records: list[dict[str, Any]] = []

    for idx, ((video_path, _), true_id, predicted_id) in enumerate(
        zip(samples, true_ids, predicted_ids)
    ):
        probs = probabilities[idx]
        order = np.argsort(probs)[::-1]
        best = float(probs[order[0]])
        second = float(probs[order[1]])

        record: dict[str, Any] = {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "player_id": parse_player_id(video_path),
            "true_label": ID_TO_LABEL[int(true_id)],
            "predicted_label": ID_TO_LABEL[int(predicted_id)],
            "correct": bool(int(true_id) == int(predicted_id)),
            "confidence": best,
            "top1_top2_margin": best - second,
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            record[f"prob_{class_name}"] = float(probs[class_id])
        records.append(record)

    prediction_frame = pd.DataFrame(records)
    prediction_frame.to_csv(
        OUTPUT_DIR / f"test_predictions_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    player_rows: list[dict[str, Any]] = []
    for player_id, group in prediction_frame.groupby("player_id", sort=True):
        row: dict[str, Any] = {
            "player_id": player_id,
            "video_count": int(len(group)),
            "correct_count": int(group["correct"].sum()),
            "accuracy": float(group["correct"].mean()),
        }
        for class_name in CLASS_NAMES:
            row[f"true_{class_name}_count"] = int((group["true_label"] == class_name).sum())
            row[f"pred_{class_name}_count"] = int((group["predicted_label"] == class_name).sum())
        player_rows.append(row)
    pd.DataFrame(player_rows).to_csv(
        OUTPUT_DIR / f"test_player_summary_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = metrics_from_ids(true_ids, predicted_ids, prefix="test_")
    with open(
        OUTPUT_DIR / f"test_metrics_{suffix}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / f"test_classification_report_{suffix}.csv",
        encoding="utf-8-sig",
    )

    save_confusion_matrix(true_ids, predicted_ids, suffix)
    return metrics


def evaluate_single_clip(
    trainer: Trainer,
    test_dataset: BasketballVideoDataset,
) -> dict[str, float]:
    print("\n" + "=" * 70)
    print("Single-Clip 测试")
    print("=" * 70)

    # 注意：Stage2 Trainer覆盖了evaluate，但predict仍使用Dataset的中心窗口逻辑
    output = trainer.predict(test_dataset, metric_key_prefix="test")
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    true_ids = np.asarray(output.label_ids)
    predicted_ids = np.argmax(logits, axis=1)
    probabilities = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=1
    ).cpu().numpy()

    metrics = save_prediction_outputs(
        true_ids,
        predicted_ids,
        probabilities,
        test_dataset.samples,
        suffix="singleclip",
    )
    print(f"Accuracy：{metrics['test_accuracy']:.4f}")
    print(f"Macro-F1：{metrics['test_macro_f1']:.4f}")
    return metrics


def evaluate_multi_clip(
    model: VideoMAEForVideoClassification,
    processor: VideoMAEImageProcessor,
    test_dataset: BasketballVideoDataset,
    num_frames: int,
    num_clips: int,
) -> dict[str, float]:
    print("\n" + "=" * 70)
    print("V8 3-Clip Center-Weighted 测试")
    print("=" * 70)

    if num_clips != 3:
        raise ValueError("V8测试固定使用3个中心附近Clip")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    true_ids: list[int] = []
    predicted_ids: list[int] = []
    all_probabilities: list[np.ndarray] = []

    for sample_index, (video_path, true_label) in enumerate(test_dataset.samples, start=1):
        print(f"[{sample_index}/{len(test_dataset.samples)}] {video_path.name}")
        frames = read_all_frames(video_path)
        clips = centered_weighted_multiclip_sample_frames(
            frames,
            num_frames=num_frames,
            window_ratio=TEST_TEMPORAL_WINDOW_RATIO,
            shift_ratio=CENTER_CLIP_SHIFT_RATIO,
        )

        clip_logits: list[torch.Tensor] = []
        with torch.inference_mode():
            for clip in clips:
                inputs = processor(clip, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                logits = model(pixel_values=pixel_values).logits[0]
                clip_logits.append(logits.detach().float().cpu())

        avg_logits = weighted_mean_logits(
            clip_logits,
            weights=CENTER_CLIP_WEIGHTS,
        )
        avg_prob = torch.softmax(avg_logits, dim=-1)
        predicted_id = int(torch.argmax(avg_prob).item())

        true_ids.append(int(true_label))
        predicted_ids.append(predicted_id)
        all_probabilities.append(avg_prob.numpy())

    true_array = np.asarray(true_ids)
    pred_array = np.asarray(predicted_ids)
    prob_array = np.stack(all_probabilities, axis=0)

    metrics = save_prediction_outputs(
        true_array,
        pred_array,
        prob_array,
        test_dataset.samples,
        suffix="multiclip",
    )
    metrics["test_num_clips"] = float(num_clips)
    metrics["test_center_clip_shift_ratio"] = float(CENTER_CLIP_SHIFT_RATIO)

    with open(
        OUTPUT_DIR / "test_metrics_multiclip.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Accuracy：{metrics['test_accuracy']:.4f}")
    print(f"Macro Precision：{metrics['test_macro_precision']:.4f}")
    print(f"Macro Recall：{metrics['test_macro_recall']:.4f}")
    print(f"Macro-F1：{metrics['test_macro_f1']:.4f}")
    return metrics


# ============================================================
# 16. 模型信息
# ============================================================
def print_model_info(model: VideoMAEForVideoClassification, pretrained_num_frames: int) -> None:
    total = sum(p.numel() for p in model.parameters())
    tubelet_size = int(model.config.tubelet_size)
    image_size = model.config.image_size
    patch_size = model.config.patch_size
    if isinstance(image_size, (list, tuple)):
        image_h, image_w = int(image_size[0]), int(image_size[1])
    else:
        image_h = image_w = int(image_size)
    if isinstance(patch_size, (list, tuple)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)

    temporal_tokens = int(model.config.num_frames) // tubelet_size
    spatial_tokens = (image_h // patch_h) * (image_w // patch_w)
    expected_tokens = temporal_tokens * spatial_tokens
    actual_tokens = int(model.videomae.embeddings.num_patches)
    pos_shape = tuple(model.videomae.embeddings.position_embeddings.shape)

    if actual_tokens != expected_tokens:
        raise RuntimeError(
            f"V8 token数量异常：expected={expected_tokens}, actual={actual_tokens}"
        )
    if pos_shape[1] != expected_tokens:
        raise RuntimeError(
            f"V8固定sin-cos位置编码长度异常：expected={expected_tokens}, shape={pos_shape}"
        )

    print("\n" + "=" * 70)
    print("VideoMAE V8模型信息")
    print("=" * 70)
    print(f"Transformers：{transformers.__version__}")
    print(f"PyTorch：{torch.__version__}")
    print(f"预训练配置帧数：{pretrained_num_frames}")
    print(f"V8输入帧数：{model.config.num_frames}")
    print(f"Tubelet Size：{tubelet_size}")
    print(f"时间Token：{temporal_tokens}")
    print(f"空间Token/时间步：{spatial_tokens}")
    print(f"总Token：{expected_tokens}")
    print(f"固定sin-cos Position Shape：{pos_shape}")
    print("位置编码策略：按32帧config重新生成固定sin-cos位置编码")
    print(f"类别数：{model.config.num_labels}")
    print(f"总参数量：{total:,}")
    print("=" * 70)


# ============================================================
# 17. 主程序
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage1_dir = OUTPUT_DIR / "stage1_classifier"
    stage2_dir = OUTPUT_DIR / "stage2_finetune"
    best_model_dir = OUTPUT_DIR / "best_model"

    if CLEAN_OLD_STAGE_OUTPUTS:
        for path in (stage1_dir, stage2_dir, best_model_dir):
            if path.exists():
                shutil.rmtree(path)

    print("=" * 70)
    print("VideoMAE V8 - Basketball Shot Type Classification")
    print("=" * 70)
    print(f"CUDA：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(
            f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    if not MODEL_NAME.exists():
        raise FileNotFoundError(f"模型目录不存在：{MODEL_NAME}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

    # --------------------------------------------------------
    # V8关键：在实例化模型之前把config.num_frames从预训练默认值改为32。
    # Hugging Face VideoMAE使用固定sin-cos位置编码，位置表会根据新的
    # num_patches = (num_frames // tubelet_size) * spatial_patches自动生成。
    # 因此这里不对“可学习位置参数”做伪插值，也不会随机初始化时间位置参数。
    # --------------------------------------------------------
    config = VideoMAEConfig.from_pretrained(MODEL_NAME)
    pretrained_num_frames = int(config.num_frames)
    config.num_frames = int(INPUT_NUM_FRAMES)
    config.num_labels = len(CLASS_NAMES)
    config.label2id = LABEL_TO_ID
    config.id2label = ID_TO_LABEL
    config.problem_type = None

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )

    print_model_info(model, pretrained_num_frames=pretrained_num_frames)
    num_frames = int(model.config.num_frames)
    if num_frames != INPUT_NUM_FRAMES:
        raise RuntimeError(
            f"V8帧数配置未生效：期望{INPUT_NUM_FRAMES}，实际{num_frames}"
        )

    print("\n" + "=" * 70)
    print("建立数据集")
    print("=" * 70)

    train_dataset = BasketballVideoDataset(
        DATA_ROOT / "train", processor, num_frames, training=True
    )
    val_dataset = BasketballVideoDataset(
        DATA_ROOT / "val", processor, num_frames, training=False
    )
    test_dataset = BasketballVideoDataset(
        DATA_ROOT / "test", processor, num_frames, training=False
    )

    print_dataset_distribution(train_dataset, "train")
    print_dataset_distribution(val_dataset, "val")
    print_dataset_distribution(test_dataset, "test")

    split_signature = save_split_manifest(
        {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
    )

    if EXPECTED_SPLIT_SIGNATURE and split_signature != EXPECTED_SPLIT_SIGNATURE:
        raise RuntimeError(
            "\nV8检测到Train/Val/Test数据划分发生变化，停止实验。\n"
            f"期望指纹：{EXPECTED_SPLIT_SIGNATURE}\n"
            f"当前指纹：{split_signature}\n"
            "请不要重新split或增删data/split中的视频。"
        )

    if PRECHECK_ALL_VIDEOS:
        precheck_all_videos(
            {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            }
        )

    # ========================================================
    # Stage 1
    # ========================================================
    freeze_for_stage1(model)
    stage1_warmup, stage1_total_steps = calculate_warmup_steps(
        len(train_dataset), STAGE1_EPOCHS
    )

    print("\n" + "=" * 70)
    print("Stage 1：冻结Backbone，只训练分类头 + fc_norm")
    print("=" * 70)
    print(f"Epoch：{STAGE1_EPOCHS}")
    print(f"LR：{STAGE1_LEARNING_RATE}")
    print("Loss：Hard Cross Entropy")
    print(f"Warmup：{stage1_warmup}/{stage1_total_steps}")

    stage1_args = make_training_args(
        stage1_dir,
        epochs=STAGE1_EPOCHS,
        learning_rate=STAGE1_LEARNING_RATE,
        warmup_steps=stage1_warmup,
        use_gradient_checkpointing=False,
        save_total_limit=2,
    )

    stage1_trainer = PlayerBalancedTrainer(
        model=model,
        args=stage1_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    stage1_result = stage1_trainer.train()
    print("\nStage 1完成")
    print(f"最佳Checkpoint：{stage1_trainer.state.best_model_checkpoint}")
    print(f"最佳验证Macro-F1：{stage1_trainer.state.best_metric}")
    save_trainer_log_and_curves(stage1_trainer, "stage1")

    model = stage1_trainer.model

    # ========================================================
    # Stage 2：全量微调 + V5分层LR + V8中心加权3-Clip Validation
    # ========================================================
    unfreeze_for_stage2(model)
    stage2_warmup, stage2_total_steps = calculate_warmup_steps(
        len(train_dataset), STAGE2_EPOCHS
    )

    print("\n" + "=" * 70)
    print("Stage 2：全量微调 + Layer-wise LR + 中心加权3-Clip Validation")
    print("=" * 70)
    print(f"最大Epoch：{STAGE2_EPOCHS}")
    print(f"Classifier LR：{CLASSIFIER_LR}")
    print(f"Top Encoder LR：{TOP_ENCODER_LR}")
    print(f"Mid Encoder LR：{MID_ENCODER_LR}")
    print(f"Bottom Encoder LR：{BOTTOM_ENCODER_LR}")
    print(f"Embedding LR：{EMBEDDING_LR}")
    print(f"Max Grad Norm：{MAX_GRAD_NORM}")
    print(f"VAL Clips：{NUM_VAL_CLIPS}")
    print(f"EarlyStopping Patience：{EARLY_STOPPING_PATIENCE}")

    stage2_args = make_training_args(
        stage2_dir,
        epochs=STAGE2_EPOCHS,
        learning_rate=CLASSIFIER_LR,
        warmup_steps=stage2_warmup,
        use_gradient_checkpointing=True,
        save_total_limit=3,
    )

    callbacks = []
    if ENABLE_EARLY_STOPPING:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        )

    stage2_trainer = LayerwiseMultiClipTrainer(
        model=model,
        args=stage2_args,
        train_dataset=train_dataset,
        # 保留eval_dataset以满足Trainer配置检查；真正evaluate已覆盖成3-Clip
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        multi_clip_eval_dataset=val_dataset,
        multi_clip_processor=processor,
        multi_clip_num_frames=num_frames,
        multi_clip_num_clips=NUM_VAL_CLIPS,
        multi_clip_window_ratio=VAL_TEMPORAL_WINDOW_RATIO,
        multi_clip_shift_ratio=CENTER_CLIP_SHIFT_RATIO,
        multi_clip_weights=CENTER_CLIP_WEIGHTS,
    )

    stage2_result = stage2_trainer.train()

    print("\n" + "=" * 70)
    print("Stage 2 最佳模型")
    print("=" * 70)
    print(f"实际结束Epoch：{stage2_trainer.state.epoch}")
    print(f"最佳Checkpoint：{stage2_trainer.state.best_model_checkpoint}")
    print(f"最佳V8 3-Clip加权VAL Macro-F1：{stage2_trainer.state.best_metric}")

    save_trainer_log_and_curves(stage2_trainer, "stage2")

    # load_best_model_at_end=True后，stage2_trainer.model已恢复最佳3-Clip VAL checkpoint
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    stage2_trainer.save_model(str(best_model_dir))
    processor.save_pretrained(str(best_model_dir))

    with open(OUTPUT_DIR / "stage1_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stage1_result.metrics, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "stage2_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stage2_result.metrics, f, ensure_ascii=False, indent=2)

    # ========================================================
    # 测试：保留Single-Clip用于对照 + V8中心加权3-Clip作为主结果
    # ========================================================
    single_metrics = evaluate_single_clip(stage2_trainer, test_dataset)
    multi_metrics = evaluate_multi_clip(
        stage2_trainer.model,
        processor,
        test_dataset,
        num_frames,
        NUM_TEST_CLIPS,
    )

    comparison = pd.DataFrame(
        [
            {
                "method": "single_clip",
                "accuracy": single_metrics["test_accuracy"],
                "macro_precision": single_metrics["test_macro_precision"],
                "macro_recall": single_metrics["test_macro_recall"],
                "macro_f1": single_metrics["test_macro_f1"],
            },
            {
                "method": f"{NUM_TEST_CLIPS}_clip_center_weighted",
                "accuracy": multi_metrics["test_accuracy"],
                "macro_precision": multi_metrics["test_macro_precision"],
                "macro_recall": multi_metrics["test_macro_recall"],
                "macro_f1": multi_metrics["test_macro_f1"],
            },
        ]
    )
    comparison.to_csv(
        OUTPUT_DIR / "single_vs_multiclip.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "version": "v8",
        "transformers_version": transformers.__version__,
        "random_seed": RANDOM_SEED,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "label_smoothing": LABEL_SMOOTHING_FACTOR,
        "loss": "hard_cross_entropy",
        "ordinal_soft_target": ENABLE_ORDINAL_SOFT_TARGET,
        "player_balance": ENABLE_PLAYER_BALANCE,
        "player_balance_power": PLAYER_BALANCE_POWER,
        "class_balance": ENABLE_CLASS_BALANCE,
        "max_grad_norm": MAX_GRAD_NORM,
        "collapse_warning_threshold": COLLAPSE_WARNING_THRESHOLD,
        "split_signature_sha256": split_signature,
        "expected_split_signature_sha256": EXPECTED_SPLIT_SIGNATURE,
        "pretrained_num_frames": pretrained_num_frames,
        "input_num_frames": num_frames,
        "tubelet_size": int(model.config.tubelet_size),
        "position_embedding_mode": "fixed_sincos_regenerated_from_config",
        "stage2_finetuning": "full_model",
        "train_temporal_min_ratio": TRAIN_TEMPORAL_MIN_RATIO,
        "train_temporal_max_ratio": TRAIN_TEMPORAL_MAX_RATIO,
        "val_num_clips": NUM_VAL_CLIPS,
        "test_num_clips": NUM_TEST_CLIPS,
        "center_clip_shift_ratio": CENTER_CLIP_SHIFT_RATIO,
        "center_clip_weights": list(CENTER_CLIP_WEIGHTS),
        "stage1_epochs": STAGE1_EPOCHS,
        "stage1_lr": STAGE1_LEARNING_RATE,
        "stage2_epochs": STAGE2_EPOCHS,
        "classifier_lr": CLASSIFIER_LR,
        "top_encoder_lr": TOP_ENCODER_LR,
        "mid_encoder_lr": MID_ENCODER_LR,
        "bottom_encoder_lr": BOTTOM_ENCODER_LR,
        "embedding_lr": EMBEDDING_LR,
        "best_stage2_checkpoint": stage2_trainer.state.best_model_checkpoint,
        "best_val_macro_f1": stage2_trainer.state.best_metric,
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "test_sample_count": len(test_dataset),
        "single_clip_test": single_metrics,
        "multi_clip_test": multi_metrics,
    }

    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("训练与测试全部完成")
    print("=" * 70)
    print(f"最佳模型：{best_model_dir}")
    print(f"Single-Clip Accuracy：{single_metrics['test_accuracy']:.4f}")
    print(f"Single-Clip Macro-F1：{single_metrics['test_macro_f1']:.4f}")
    print(f"{NUM_TEST_CLIPS}-Clip Accuracy：{multi_metrics['test_accuracy']:.4f}")
    print(f"{NUM_TEST_CLIPS}-Clip Macro-F1：{multi_metrics['test_macro_f1']:.4f}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
