from __future__ import annotations

import json
import math
import random
import shutil
import time
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
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    set_seed,
)


# ============================================================
# 1. 路径与类别
# ============================================================
PROJECT_ROOT = Path(r"F:\BasketballVideoMAE")
DATA_ROOT = PROJECT_ROOT / "data" / "split"

# 建议使用新目录，避免覆盖之前的 baseline，方便对比实验
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videomae_twostage"
MODEL_NAME = PROJECT_ROOT / "models" / "videomae-base"

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
# 2. 训练超参数：B + C
# ============================================================
RANDOM_SEED = 42
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.10

# B：降低过拟合/过度自信
LABEL_SMOOTHING_FACTOR = 0.10

# C：两阶段训练
STAGE1_EPOCHS = 5
STAGE1_LEARNING_RATE = 1e-4       # 只训练分类头/最终归一化层

STAGE2_EPOCHS = 35
STAGE2_LEARNING_RATE = 2e-5       # 解冻全部参数后，小学习率微调

ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_THRESHOLD = 0.0

# D：测试集 Multi-Clip
NUM_TEST_CLIPS = 5

# 视频检查
PRECHECK_ALL_VIDEOS = True
PRECHECK_PRINT_EVERY = 10
MAX_VIDEO_READ_RETRIES = 3
VIDEO_RETRY_WAIT_SECONDS = 0.5
MAX_ALLOWED_FRAME_DIFFERENCE = 3
FRAME_COUNT_TOLERANCE_RATIO = 0.02

# 每次正式重新训练时清掉这个新实验目录中的旧结果
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
# 4. 基础工具
# ============================================================
def get_allowed_frame_difference(expected_frame_count: int) -> int:
    if expected_frame_count <= 0:
        return MAX_ALLOWED_FRAME_DIFFERENCE
    ratio_difference = int(
        np.ceil(expected_frame_count * FRAME_COUNT_TOLERANCE_RATIO)
    )
    return max(MAX_ALLOWED_FRAME_DIFFERENCE, ratio_difference)


def check_video_decodable(video_path: Path) -> tuple[bool, str]:
    """完整解码检查，不把所有帧存到内存。"""
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
    """完整读取视频，失败时自动重新打开，最多重试 max_retries 次。"""
    last_error = "未知错误"

    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            last_error = "无法打开视频"
            cap.release()
            print(f"[视频读取警告] {attempt}/{max_retries}：{video_path}")
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
                print(f"[视频读取警告] {last_error}\n文件：{video_path}")
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
# 5. 时间采样
# ============================================================
def random_segment_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
) -> list[np.ndarray]:
    """TRAIN：把整段视频分成 num_frames 段，每段随机抽一帧。"""
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total < num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        return [frames[i] for i in indices]

    boundaries = np.linspace(0, total, num=num_frames + 1)
    indices: list[int] = []
    for i in range(num_frames):
        start = int(np.floor(boundaries[i]))
        end = int(np.floor(boundaries[i + 1]))
        end = max(end, start + 1)
        end = min(end, total)
        indices.append(random.randint(start, end - 1))
    return [frames[i] for i in indices]


def center_segment_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
) -> list[np.ndarray]:
    """VAL/单Clip TEST：每个时间段固定取中心帧。"""
    total = len(frames)
    if total <= 0:
        raise RuntimeError("视频没有有效帧")

    if total < num_frames:
        indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
        return [frames[i] for i in indices]

    boundaries = np.linspace(0, total, num=num_frames + 1)
    indices: list[int] = []
    for i in range(num_frames):
        start = int(np.floor(boundaries[i]))
        end = int(np.floor(boundaries[i + 1]))
        end = max(end, start + 1)
        end = min(end, total)
        indices.append((start + end - 1) // 2)
    return [frames[i] for i in indices]


def multi_clip_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
    num_clips: int = 5,
) -> list[list[np.ndarray]]:
    """
    D：从同一视频构造多个时间采样 Clip。
    例如5个Clip在每个时间段分别取约10/30/50/70/90%位置。
    """
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


# ============================================================
# 6. Dataset
# ============================================================
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

        for class_name in CLASS_NAMES:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"缺少类别文件夹：{class_dir}")

            label_id = LABEL_TO_ID[class_name]
            for video_path in sorted(class_dir.rglob("*")):
                if (
                    video_path.is_file()
                    and video_path.suffix.lower() in VIDEO_EXTENSIONS
                ):
                    self.samples.append((video_path, label_id))

        if not self.samples:
            raise RuntimeError(f"数据集中没有视频：{root_dir}")

        print(f"{root_dir.name}：{len(self.samples)} 个视频")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        video_path, label = self.samples[index]
        frames = read_all_frames(video_path)

        if self.training:
            frames = random_segment_sample_frames(frames, self.num_frames)
        else:
            frames = center_segment_sample_frames(frames, self.num_frames)

        # 轻量增强：左右镜像不改变一段/1.5段/二段标签
        if self.training and random.random() < 0.5:
            frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]

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


def print_dataset_distribution(
    dataset: BasketballVideoDataset,
    split_name: str,
) -> None:
    labels = [label for _, label in dataset.samples]
    print("\n" + "-" * 70)
    print(f"{split_name.upper()} 类别分布")
    print("-" * 70)
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = labels.count(class_id)
        print(f"{class_name:<24}{count:>5} ({count / len(labels):.2%})")


def precheck_all_videos(
    datasets: dict[str, BasketballVideoDataset],
) -> None:
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
            elif (
                current == 1
                or current % PRECHECK_PRINT_EVERY == 0
                or current == total
            ):
                print(f"[{current}/{total}] OK：{video_path.name}")

    if failed:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "video_precheck_failures.csv"
        pd.DataFrame(failed).to_csv(path, index=False, encoding="utf-8-sig")
        raise RuntimeError(f"存在异常视频，报告已保存：{path}")

    print(f"全部 {total} 个视频均通过完整解码检查。")


# ============================================================
# 7. 指标
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
# 8. 两阶段训练辅助函数
# ============================================================
def freeze_for_stage1(model: VideoMAEForVideoClassification) -> None:
    """Stage 1：冻结全部参数，仅解冻 classifier 和 fc_norm（若存在）。"""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if hasattr(model, "fc_norm") and model.fc_norm is not None:
        for parameter in model.fc_norm.parameters():
            parameter.requires_grad = True

    # Stage1 Backbone 已冻结，不需要 gradient checkpointing
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    print_trainable_parameters(model, "Stage 1")


def unfreeze_for_stage2(model: VideoMAEForVideoClassification) -> None:
    """Stage 2：解冻全部参数。"""
    for parameter in model.parameters():
        parameter.requires_grad = True

    model.gradient_checkpointing_enable()
    print_trainable_parameters(model, "Stage 2")


def print_trainable_parameters(
    model: VideoMAEForVideoClassification,
    stage_name: str,
) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 70)
    print(f"{stage_name} 参数状态")
    print("=" * 70)
    print(f"模型总参数：{total:,}")
    print(f"可训练参数：{trainable:,}")
    print(f"可训练比例：{trainable / total:.4%}")


def print_model_info(model: VideoMAEForVideoClassification) -> None:
    print("\n" + "=" * 70)
    print("模型信息")
    print("=" * 70)
    print(f"Transformers：{transformers.__version__}")
    print(f"PyTorch：{torch.__version__}")
    print(f"model_type：{model.config.model_type}")
    print(f"num_frames：{model.config.num_frames}")
    print(f"hidden_size：{model.config.hidden_size}")
    print(f"num_hidden_layers：{model.config.num_hidden_layers}")
    print(f"num_attention_heads：{model.config.num_attention_heads}")
    print(f"num_labels：{model.config.num_labels}")

    total = sum(p.numel() for p in model.parameters())
    classifier = sum(
        p.numel() for name, p in model.named_parameters() if "classifier" in name
    )
    print(f"总参数量：{total:,}")
    print(f"分类头参数量：{classifier:,}")

    print("\nEncoder第一层权重检查：")
    for name, param in model.named_parameters():
        if "videomae.encoder.layer.0" in name and name.endswith(".weight"):
            t = param.detach().float().cpu()
            print(name)
            print(f"mean: {t.mean().item():.8f}")
            print(f"std:  {t.std().item():.8f}")
            break


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


def save_trainer_log_and_curves(trainer: Trainer, prefix: str) -> None:
    logs = trainer.state.log_history
    if not logs:
        return

    pd.DataFrame(logs).to_csv(
        OUTPUT_DIR / f"{prefix}_training_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_logs = [
        x for x in logs
        if "loss" in x and "eval_loss" not in x and "step" in x
    ]
    if train_logs:
        plt.figure(figsize=(8, 5))
        plt.plot([x["step"] for x in train_logs], [x["loss"] for x in train_logs])
        plt.xlabel("Training Step")
        plt.ylabel("Train Loss")
        plt.title(f"{prefix} Training Loss")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{prefix}_train_loss.png", dpi=200)
        plt.close()

    eval_logs = [x for x in logs if "eval_macro_f1" in x and "epoch" in x]
    if eval_logs:
        plt.figure(figsize=(8, 5))
        plt.plot(
            [x["epoch"] for x in eval_logs],
            [x["eval_macro_f1"] for x in eval_logs],
            marker="o",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Macro-F1")
        plt.title(f"{prefix} Validation Macro-F1")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{prefix}_val_macro_f1.png", dpi=200)
        plt.close()


# ============================================================
# 9. 测试结果保存
# ============================================================
def save_confusion_matrix(
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
    suffix: str,
) -> None:
    matrix = confusion_matrix(
        true_ids,
        predicted_ids,
        labels=list(range(len(CLASS_NAMES))),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    display.plot(ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {suffix}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{suffix}.png", dpi=200)
    plt.close(fig)

    normalized = confusion_matrix(
        true_ids,
        predicted_ids,
        labels=list(range(len(CLASS_NAMES))),
        normalize="true",
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(normalized, display_labels=CLASS_NAMES)
    display.plot(ax=ax, values_format=".2f")
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
            "true_label": ID_TO_LABEL[int(true_id)],
            "predicted_label": ID_TO_LABEL[int(predicted_id)],
            "correct": bool(int(true_id) == int(predicted_id)),
            "confidence": best,
            "top1_top2_margin": best - second,
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            record[f"prob_{class_name}"] = float(probs[class_id])
        records.append(record)

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / f"test_predictions_{suffix}.csv",
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
    """保留单Clip测试，用于和旧实验公平比较。"""
    print("\n" + "=" * 70)
    print("Single-Clip 测试")
    print("=" * 70)

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
    """D：对测试集逐视频做多个时间Clip并平均概率。"""
    print("\n" + "=" * 70)
    print(f"{num_clips}-Clip 测试")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    true_ids: list[int] = []
    predicted_ids: list[int] = []
    all_probabilities: list[np.ndarray] = []

    for sample_index, (video_path, true_label) in enumerate(
        test_dataset.samples, start=1
    ):
        print(f"[{sample_index}/{len(test_dataset.samples)}] {video_path.name}")
        frames = read_all_frames(video_path)
        clips = multi_clip_sample_frames(
            frames,
            num_frames=num_frames,
            num_clips=num_clips,
        )

        clip_probs: list[torch.Tensor] = []
        with torch.inference_mode():
            for clip in clips:
                inputs = processor(clip, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                logits = model(pixel_values=pixel_values).logits
                clip_probs.append(torch.softmax(logits, dim=-1)[0].cpu())

        avg_prob = torch.stack(clip_probs, dim=0).mean(dim=0)
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
# 10. 主程序
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage1_dir = OUTPUT_DIR / "stage1_classifier"
    stage2_dir = OUTPUT_DIR / "stage2_finetune"
    best_model_dir = OUTPUT_DIR / "best_model"

    if CLEAN_OLD_STAGE_OUTPUTS:
        for path in (stage1_dir, stage2_dir, best_model_dir):
            if path.exists():
                shutil.rmtree(path)

    print("=" * 70)
    print("设备检查")
    print("=" * 70)
    print(f"Transformers：{transformers.__version__}")
    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(
            f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    if not MODEL_NAME.exists():
        raise FileNotFoundError(f"模型目录不存在：{MODEL_NAME}")

    # --------------------------------------------------------
    # Processor / Model
    # --------------------------------------------------------
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
        ignore_mismatched_sizes=True,
    )

    print_model_info(model)
    num_frames = int(model.config.num_frames)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
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

    if PRECHECK_ALL_VIDEOS:
        precheck_all_videos(
            {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            }
        )

    # ========================================================
    # Stage 1：冻结 Backbone，仅训练分类头 + fc_norm
    # ========================================================
    freeze_for_stage1(model)
    stage1_warmup, stage1_total_steps = calculate_warmup_steps(
        len(train_dataset), STAGE1_EPOCHS
    )

    print("\n" + "=" * 70)
    print("Stage 1：冻结 Backbone，训练分类头")
    print("=" * 70)
    print(f"Epoch：{STAGE1_EPOCHS}")
    print(f"LR：{STAGE1_LEARNING_RATE}")
    print(f"Label Smoothing：{LABEL_SMOOTHING_FACTOR}")
    print(f"Warmup Steps：{stage1_warmup}/{stage1_total_steps}")

    stage1_args = make_training_args(
        stage1_dir,
        epochs=STAGE1_EPOCHS,
        learning_rate=STAGE1_LEARNING_RATE,
        warmup_steps=stage1_warmup,
        use_gradient_checkpointing=False,
        save_total_limit=2,
    )

    stage1_trainer = Trainer(
        model=model,
        args=stage1_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    stage1_result = stage1_trainer.train()

    print("\nStage 1 完成")
    print(f"最佳Checkpoint：{stage1_trainer.state.best_model_checkpoint}")
    print(f"最佳验证Macro-F1：{stage1_trainer.state.best_metric}")
    save_trainer_log_and_curves(stage1_trainer, "stage1")

    # Trainer在load_best_model_at_end=True时已经把Stage1最佳权重重新装回model
    model = stage1_trainer.model

    # ========================================================
    # Stage 2：解冻全部参数，小学习率微调
    # ========================================================
    unfreeze_for_stage2(model)
    stage2_warmup, stage2_total_steps = calculate_warmup_steps(
        len(train_dataset), STAGE2_EPOCHS
    )

    print("\n" + "=" * 70)
    print("Stage 2：解冻全部参数进行微调")
    print("=" * 70)
    print(f"最大Epoch：{STAGE2_EPOCHS}")
    print(f"LR：{STAGE2_LEARNING_RATE}")
    print(f"Label Smoothing：{LABEL_SMOOTHING_FACTOR}")
    print(f"Warmup Steps：{stage2_warmup}/{stage2_total_steps}")
    print(f"EarlyStopping Patience：{EARLY_STOPPING_PATIENCE}")

    stage2_args = make_training_args(
        stage2_dir,
        epochs=STAGE2_EPOCHS,
        learning_rate=STAGE2_LEARNING_RATE,
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

    stage2_trainer = Trainer(
        model=model,
        args=stage2_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    stage2_result = stage2_trainer.train()

    print("\n" + "=" * 70)
    print("Stage 2 最佳模型")
    print("=" * 70)
    print(f"实际结束Epoch：{stage2_trainer.state.epoch}")
    print(f"最佳Checkpoint：{stage2_trainer.state.best_model_checkpoint}")
    print(f"最佳验证Macro-F1：{stage2_trainer.state.best_metric}")

    save_trainer_log_and_curves(stage2_trainer, "stage2")

    # --------------------------------------------------------
    # 保存最终最佳模型
    # --------------------------------------------------------
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    stage2_trainer.save_model(str(best_model_dir))
    processor.save_pretrained(str(best_model_dir))

    with open(OUTPUT_DIR / "stage1_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stage1_result.metrics, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "stage2_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stage2_result.metrics, f, ensure_ascii=False, indent=2)

    summary = {
        "transformers_version": transformers.__version__,
        "random_seed": RANDOM_SEED,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage1_learning_rate": STAGE1_LEARNING_RATE,
        "stage1_best_checkpoint": stage1_trainer.state.best_model_checkpoint,
        "stage1_best_macro_f1": stage1_trainer.state.best_metric,
        "stage2_max_epochs": STAGE2_EPOCHS,
        "stage2_actual_final_epoch": stage2_trainer.state.epoch,
        "stage2_learning_rate": STAGE2_LEARNING_RATE,
        "stage2_best_checkpoint": stage2_trainer.state.best_model_checkpoint,
        "stage2_best_macro_f1": stage2_trainer.state.best_metric,
        "label_smoothing_factor": LABEL_SMOOTHING_FACTOR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "num_test_clips": NUM_TEST_CLIPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    }
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ========================================================
    # 测试：Single-Clip + Multi-Clip
    # ========================================================
    single_metrics = evaluate_single_clip(stage2_trainer, test_dataset)

    multi_metrics = evaluate_multi_clip(
        stage2_trainer.model,
        processor,
        test_dataset,
        num_frames=num_frames,
        num_clips=NUM_TEST_CLIPS,
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
                "method": f"{NUM_TEST_CLIPS}_clip",
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

    print("\n" + "=" * 70)
    print("全部训练与测试完成")
    print("=" * 70)
    print(f"最终最佳模型：{best_model_dir}")
    print(f"Single-Clip Accuracy：{single_metrics['test_accuracy']:.4f}")
    print(f"Single-Clip Macro-F1：{single_metrics['test_macro_f1']:.4f}")
    print(f"{NUM_TEST_CLIPS}-Clip Accuracy：{multi_metrics['test_accuracy']:.4f}")
    print(f"{NUM_TEST_CLIPS}-Clip Macro-F1：{multi_metrics['test_macro_f1']:.4f}")
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
