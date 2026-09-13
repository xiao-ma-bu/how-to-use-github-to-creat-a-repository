from __future__ import annotations

import gc
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

from safetensors.torch import load_file as safe_load_file

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

from transformers.trainer_utils import (
    get_last_checkpoint,
)


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(
    r"F:\BasketballVideoMAE"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "split"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "videomae_baseline"
)

MODEL_NAME = (
    PROJECT_ROOT
    / "models"
    / "videomae-base"
)


# ============================================================
# 2. 分类类别
# ============================================================

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

LABEL_TO_ID = {
    class_name: index
    for index, class_name
    in enumerate(CLASS_NAMES)
}

ID_TO_LABEL = {
    index: class_name
    for class_name, index
    in LABEL_TO_ID.items()
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# 3. 训练超参数
# ============================================================

RANDOM_SEED = 42

# 最大训练轮数
NUM_EPOCHS = 40

TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

# 有效 Batch Size ≈ 1 × 8 = 8
GRADIENT_ACCUMULATION_STEPS = 8


# ============================================================
# 学习率
# ============================================================

# 全量微调阶段使用更小的学习率，
# 避免小数据集快速破坏VideoMAE预训练特征。
LEARNING_RATE = 2e-5

# ============================================================
# Label Smoothing
# ============================================================

# 用于抑制模型过度自信。
#
# 例如原本可能：
#
# 错误预测
# confidence = 0.999
#
# Label Smoothing可以适当缓解这种情况。
label_smoothing_factor = 0.1

WEIGHT_DECAY = 0.05


# ------------------------------------------------------------
# Warmup
#
# 注意：
# 这里只把 0.10 作为我们自己计算 warmup_steps 的比例。
#
# 不再向 TrainingArguments 传入 warmup_ratio，
# 因此不会再触发：
#
# warmup_ratio is deprecated
# ------------------------------------------------------------

WARMUP_RATIO = 0.10


# ============================================================
# 4. Early Stopping
# ============================================================

ENABLE_EARLY_STOPPING = True

# 连续 8 个 Epoch
# Validation Macro-F1 没有改善就停止
EARLY_STOPPING_PATIENCE = 8

EARLY_STOPPING_THRESHOLD = 0.0


# ============================================================
# 5. 视频检查参数
# ============================================================

# 建议正式训练时保持 True
PRECHECK_ALL_VIDEOS = True

PRECHECK_PRINT_EVERY = 10

# 训练期间视频偶发读取失败最多尝试 3 次
MAX_VIDEO_READ_RETRIES = 3

VIDEO_RETRY_WAIT_SECONDS = 0.5

# 视频声明帧数与实际帧数允许的最小差值
MAX_ALLOWED_FRAME_DIFFERENCE = 3

# 对较长视频允许约 2% 的 metadata 偏差
FRAME_COUNT_TOLERANCE_RATIO = 0.02


# ============================================================
# 6. 是否继续旧 checkpoint
# ============================================================

# 如果重新清洗过标签或者重新 split：
#
# 一定 False
#
# 如果仅仅是训练意外中断，
# 且数据完全没有变化，
# 可以改 True。
RESUME_FROM_CHECKPOINT = False


# ============================================================
# 7. 保存 best_model 设置
# ============================================================

REMOVE_OLD_BEST_MODEL_BEFORE_SAVE = True


# ============================================================
# 8. 是否检查预训练 Encoder
# ============================================================

# 强烈建议保持 True。
#
# 每次训练开始前，
# 会检查预训练 checkpoint
# 到当前 VideoMAE 分类模型之间
# Encoder 权重究竟加载了多少。
CHECK_PRETRAINED_ENCODER = True


# ============================================================
# 9. Encoder 权重比较参数
# ============================================================

# allclose 判断阈值
ENCODER_COMPARE_RTOL = 1e-5
ENCODER_COMPARE_ATOL = 1e-7


# ============================================================
# 10. 固定随机种子
# ============================================================

set_seed(
    RANDOM_SEED
)

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# 11. 加载本地 HuggingFace checkpoint
# ============================================================

def load_local_checkpoint_state_dict(
    model_dir: Path,
) -> dict[str, torch.Tensor]:
    """
    读取本地 HuggingFace checkpoint。

    支持：

    1. model.safetensors
    2. model.safetensors.index.json
    3. pytorch_model.bin
    4. pytorch_model.bin.index.json
    """

    model_dir = Path(
        model_dir
    )

    # ========================================================
    # 1. 单文件 safetensors
    # ========================================================

    safetensor_path = (
        model_dir
        / "model.safetensors"
    )

    if safetensor_path.exists():

        print(
            "检测到权重文件："
            f"{safetensor_path.name}"
        )

        return safe_load_file(
            str(
                safetensor_path
            ),
            device="cpu",
        )


    # ========================================================
    # 2. 分片 safetensors
    # ========================================================

    safetensor_index_path = (
        model_dir
        / "model.safetensors.index.json"
    )

    if safetensor_index_path.exists():

        print(
            "检测到分片权重："
            f"{safetensor_index_path.name}"
        )

        with open(
            safetensor_index_path,
            "r",
            encoding="utf-8",
        ) as file:

            index_data = json.load(
                file
            )

        weight_map = (
            index_data[
                "weight_map"
            ]
        )

        shard_names = sorted(
            set(
                weight_map.values()
            )
        )

        state_dict: dict[
            str,
            torch.Tensor
        ] = {}

        for shard_name in shard_names:

            shard_path = (
                model_dir
                / shard_name
            )

            print(
                f"读取分片："
                f"{shard_name}"
            )

            shard_state = safe_load_file(
                str(
                    shard_path
                ),
                device="cpu",
            )

            state_dict.update(
                shard_state
            )

        return state_dict


    # ========================================================
    # 3. 单文件 pytorch_model.bin
    # ========================================================

    bin_path = (
        model_dir
        / "pytorch_model.bin"
    )

    if bin_path.exists():

        print(
            "检测到权重文件："
            f"{bin_path.name}"
        )

        try:

            checkpoint = torch.load(
                bin_path,
                map_location="cpu",
                weights_only=True,
            )

        except TypeError:

            checkpoint = torch.load(
                bin_path,
                map_location="cpu",
            )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and
            "state_dict"
            in checkpoint
            and
            isinstance(
                checkpoint[
                    "state_dict"
                ],
                dict,
            )
        ):

            checkpoint = (
                checkpoint[
                    "state_dict"
                ]
            )

        return checkpoint


    # ========================================================
    # 4. 分片 pytorch_model.bin
    # ========================================================

    bin_index_path = (
        model_dir
        / "pytorch_model.bin.index.json"
    )

    if bin_index_path.exists():

        print(
            "检测到分片权重："
            f"{bin_index_path.name}"
        )

        with open(
            bin_index_path,
            "r",
            encoding="utf-8",
        ) as file:

            index_data = json.load(
                file
            )

        weight_map = (
            index_data[
                "weight_map"
            ]
        )

        shard_names = sorted(
            set(
                weight_map.values()
            )
        )

        state_dict: dict[
            str,
            torch.Tensor
        ] = {}

        for shard_name in shard_names:

            shard_path = (
                model_dir
                / shard_name
            )

            print(
                f"读取分片："
                f"{shard_name}"
            )

            try:

                shard_state = torch.load(
                    shard_path,
                    map_location="cpu",
                    weights_only=True,
                )

            except TypeError:

                shard_state = torch.load(
                    shard_path,
                    map_location="cpu",
                )

            if (
                isinstance(
                    shard_state,
                    dict,
                )
                and
                "state_dict"
                in shard_state
            ):

                shard_state = (
                    shard_state[
                        "state_dict"
                    ]
                )

            state_dict.update(
                shard_state
            )

        return state_dict


    raise FileNotFoundError(
        "\n没有找到 VideoMAE 权重文件。\n"
        f"目录：{model_dir}\n"
        "应至少存在以下之一：\n"
        "model.safetensors\n"
        "model.safetensors.index.json\n"
        "pytorch_model.bin\n"
        "pytorch_model.bin.index.json"
    )


# ============================================================
# 12. 打印 Transformers 版本
# ============================================================

def print_transformers_version() -> None:

    print("\n" + "=" * 70)
    print("Transformers 环境")
    print("=" * 70)

    print(
        "Transformers版本："
        f"{transformers.__version__}"
    )

    print(
        "PyTorch版本："
        f"{torch.__version__}"
    )

    print("=" * 70)


# ============================================================
# 13. 打印 VideoMAE Config
# ============================================================

def print_videomae_config(
    model: VideoMAEForVideoClassification,
) -> None:

    config = (
        model.config
    )

    print("\n" + "=" * 70)
    print("VideoMAE Config")
    print("=" * 70)

    config_fields = [
        (
            "模型类型",
            getattr(
                config,
                "model_type",
                None,
            ),
        ),
        (
            "输入帧数",
            getattr(
                config,
                "num_frames",
                None,
            ),
        ),
        (
            "图像尺寸",
            getattr(
                config,
                "image_size",
                None,
            ),
        ),
        (
            "Patch Size",
            getattr(
                config,
                "patch_size",
                None,
            ),
        ),
        (
            "Tubelet Size",
            getattr(
                config,
                "tubelet_size",
                None,
            ),
        ),
        (
            "Hidden Size",
            getattr(
                config,
                "hidden_size",
                None,
            ),
        ),
        (
            "Encoder层数",
            getattr(
                config,
                "num_hidden_layers",
                None,
            ),
        ),
        (
            "Attention Heads",
            getattr(
                config,
                "num_attention_heads",
                None,
            ),
        ),
        (
            "Intermediate Size",
            getattr(
                config,
                "intermediate_size",
                None,
            ),
        ),
        (
            "类别数量",
            getattr(
                config,
                "num_labels",
                None,
            ),
        ),
    ]

    for key, value in config_fields:

        print(
            f"{key}："
            f"{value}"
        )

    print("=" * 70)


# ============================================================
# 14. 模型参数统计
# ============================================================

def print_model_parameter_statistics(
    model: VideoMAEForVideoClassification,
) -> None:

    print("\n" + "=" * 70)
    print("模型参数统计")
    print("=" * 70)

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    classifier_parameters = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if "classifier" in name
    )

    encoder_parameters = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if name.startswith(
            "videomae.encoder"
        )
    )

    print(
        f"模型总参数量："
        f"{total_parameters:,}"
    )

    print(
        f"可训练参数量："
        f"{trainable_parameters:,}"
    )

    print(
        f"Encoder参数量："
        f"{encoder_parameters:,}"
    )

    print(
        f"分类头参数量："
        f"{classifier_parameters:,}"
    )

    if total_parameters > 0:

        print(
            "Encoder占总参数比例："
            f"{encoder_parameters / total_parameters:.4%}"
        )

    print("=" * 70)


# ============================================================
# 15. 检查 Encoder 预训练权重加载情况
# ============================================================

def check_encoder_loading(
    model_dir: Path,
    model: VideoMAEForVideoClassification,
) -> None:
    """
    不仅检查参数名称，
    还实际比较 checkpoint Tensor
    与当前 model Tensor 的数值。

    从而判断：

    当前模型中的 Encoder 参数
    是否真正继承自预训练 checkpoint。
    """

    print("\n" + "=" * 70)
    print("预训练 Encoder 权重加载检查")
    print("=" * 70)

    checkpoint = (
        load_local_checkpoint_state_dict(
            model_dir
        )
    )

    model_state_dict = (
        model.state_dict()
    )

    checkpoint_encoder_keys = sorted(
        [
            key
            for key
            in checkpoint.keys()
            if key.startswith(
                "videomae.encoder"
            )
        ]
    )

    model_encoder_keys = sorted(
        [
            key
            for key
            in model_state_dict.keys()
            if key.startswith(
                "videomae.encoder"
            )
        ]
    )

    checkpoint_encoder_key_set = set(
        checkpoint_encoder_keys
    )

    model_encoder_key_set = set(
        model_encoder_keys
    )

    # ========================================================
    # 名称完全相同
    # ========================================================

    matched_keys = sorted(
        checkpoint_encoder_key_set
        &
        model_encoder_key_set
    )

    # ========================================================
    # checkpoint有、当前模型没有
    # ========================================================

    checkpoint_only_keys = sorted(
        checkpoint_encoder_key_set
        -
        model_encoder_key_set
    )

    # ========================================================
    # 当前模型有、checkpoint没有
    #
    # 这些参数通常就是 MISSING，
    # 会被重新初始化。
    # ========================================================

    model_only_keys = sorted(
        model_encoder_key_set
        -
        checkpoint_encoder_key_set
    )


    # ========================================================
    # 数值逐个比较
    # ========================================================

    shape_matched_count = 0

    value_matched_count = 0

    value_mismatched_count = 0

    shape_mismatched_count = 0


    checkpoint_encoder_numel = 0

    name_matched_numel = 0

    value_matched_numel = 0


    comparison_records: list[
        dict[str, Any]
    ] = []


    # --------------------------------------------------------
    # checkpoint Encoder 总元素数
    # --------------------------------------------------------

    for key in checkpoint_encoder_keys:

        tensor = checkpoint[
            key
        ]

        checkpoint_encoder_numel += (
            tensor.numel()
        )


    # --------------------------------------------------------
    # 对名称相同参数比较
    # --------------------------------------------------------

    for key in matched_keys:

        checkpoint_tensor = (
            checkpoint[
                key
            ]
            .detach()
            .cpu()
        )

        model_tensor = (
            model_state_dict[
                key
            ]
            .detach()
            .cpu()
        )

        record: dict[
            str,
            Any
        ] = {
            "parameter":
                key,

            "checkpoint_shape":
                str(
                    tuple(
                        checkpoint_tensor.shape
                    )
                ),

            "model_shape":
                str(
                    tuple(
                        model_tensor.shape
                    )
                ),

            "numel":
                int(
                    checkpoint_tensor.numel()
                ),
        }

        # ----------------------------------------------------
        # Shape 是否一致
        # ----------------------------------------------------

        if (
            checkpoint_tensor.shape
            !=
            model_tensor.shape
        ):

            shape_mismatched_count += 1

            record[
                "status"
            ] = "SHAPE_MISMATCH"

            comparison_records.append(
                record
            )

            continue

        shape_matched_count += 1

        name_matched_numel += (
            checkpoint_tensor.numel()
        )


        # ----------------------------------------------------
        # 为避免 dtype 差异影响判断，
        # 转 float32 后 allclose
        # ----------------------------------------------------

        checkpoint_compare = (
            checkpoint_tensor
            .to(
                dtype=torch.float32
            )
        )

        model_compare = (
            model_tensor
            .to(
                dtype=torch.float32
            )
        )


        value_same = torch.allclose(
            checkpoint_compare,
            model_compare,
            rtol=ENCODER_COMPARE_RTOL,
            atol=ENCODER_COMPARE_ATOL,
        )


        if value_same:

            value_matched_count += 1

            value_matched_numel += (
                checkpoint_tensor.numel()
            )

            record[
                "status"
            ] = "VALUE_MATCH"

        else:

            value_mismatched_count += 1

            difference = torch.max(
                torch.abs(
                    checkpoint_compare
                    -
                    model_compare
                )
            ).item()

            record[
                "status"
            ] = "VALUE_MISMATCH"

            record[
                "max_abs_difference"
            ] = difference


        comparison_records.append(
            record
        )


    # ========================================================
    # checkpoint-only 参数也写报告
    # ========================================================

    for key in checkpoint_only_keys:

        tensor = checkpoint[
            key
        ]

        comparison_records.append(
            {
                "parameter":
                    key,

                "checkpoint_shape":
                    str(
                        tuple(
                            tensor.shape
                        )
                    ),

                "model_shape":
                    "",

                "numel":
                    int(
                        tensor.numel()
                    ),

                "status":
                    "CHECKPOINT_ONLY",
            }
        )


    # ========================================================
    # model-only 参数
    # ========================================================

    for key in model_only_keys:

        tensor = model_state_dict[
            key
        ]

        comparison_records.append(
            {
                "parameter":
                    key,

                "checkpoint_shape":
                    "",

                "model_shape":
                    str(
                        tuple(
                            tensor.shape
                        )
                    ),

                "numel":
                    int(
                        tensor.numel()
                    ),

                "status":
                    "MODEL_ONLY_NEW_INIT",
            }
        )


    # ========================================================
    # 输出统计
    # ========================================================

    checkpoint_key_count = len(
        checkpoint_encoder_keys
    )

    model_key_count = len(
        model_encoder_keys
    )

    name_match_count = len(
        matched_keys
    )


    print(
        "Checkpoint Encoder Tensor数量："
        f"{checkpoint_key_count}"
    )

    print(
        "当前模型 Encoder Tensor数量："
        f"{model_key_count}"
    )

    print(
        "参数名称匹配数量："
        f"{name_match_count}"
    )

    if checkpoint_key_count > 0:

        print(
            "按Tensor数量计算的名称匹配率："
            f"{name_match_count / checkpoint_key_count:.2%}"
        )


    print(
        "Shape匹配数量："
        f"{shape_matched_count}"
    )

    print(
        "Shape不匹配数量："
        f"{shape_mismatched_count}"
    )

    print(
        "数值真正匹配数量："
        f"{value_matched_count}"
    )

    print(
        "名称相同但数值不同数量："
        f"{value_mismatched_count}"
    )


    # ========================================================
    # 参数量加权统计
    # ========================================================

    if checkpoint_encoder_numel > 0:

        name_numel_ratio = (
            name_matched_numel
            / checkpoint_encoder_numel
        )

        value_numel_ratio = (
            value_matched_numel
            / checkpoint_encoder_numel
        )

        print(
            "\n按实际参数元素数量统计："
        )

        print(
            "Checkpoint Encoder总参数元素："
            f"{checkpoint_encoder_numel:,}"
        )

        print(
            "名称+Shape匹配参数元素："
            f"{name_matched_numel:,}"
        )

        print(
            "名称+Shape匹配比例："
            f"{name_numel_ratio:.6%}"
        )

        print(
            "数值真正继承参数元素："
            f"{value_matched_numel:,}"
        )

        print(
            "Encoder预训练权重实际继承比例："
            f"{value_numel_ratio:.6%}"
        )


    # ========================================================
    # 输出 checkpoint 中未匹配 Encoder 参数
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "Checkpoint中存在、"
        "但当前模型没有的Encoder参数："
    )

    print("-" * 70)

    if not checkpoint_only_keys:

        print(
            "无"
        )

    else:

        for key in checkpoint_only_keys:

            print(
                f"  {key}"
            )


    # ========================================================
    # 输出当前模型新初始化 Encoder 参数
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "当前模型存在、"
        "但Checkpoint没有的Encoder参数："
    )

    print("-" * 70)

    if not model_only_keys:

        print(
            "无"
        )

    else:

        for key in model_only_keys:

            print(
                f"  {key}"
            )


    # ========================================================
    # 如果名称一样但权重数值没加载
    # ========================================================

    mismatch_records = [
        record
        for record
        in comparison_records
        if record[
            "status"
        ] in {
            "VALUE_MISMATCH",
            "SHAPE_MISMATCH",
        }
    ]

    print("\n" + "-" * 70)

    print(
        "名称相同但未正确继承的Encoder参数："
    )

    print("-" * 70)

    if not mismatch_records:

        print(
            "无"
        )

    else:

        for record in mismatch_records:

            print(
                f"  {record['parameter']} "
                f"[{record['status']}]"
            )


    # ========================================================
    # 保存详细报告
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "encoder_loading_report.csv"
    )

    pd.DataFrame(
        comparison_records
    ).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nEncoder详细检查结果已保存："
    )

    print(
        report_path
    )


    # ========================================================
    # 简单自动判断
    # ========================================================

    if checkpoint_encoder_numel > 0:

        inherited_ratio = (
            value_matched_numel
            / checkpoint_encoder_numel
        )

        print("\n" + "=" * 70)
        print("Encoder加载自动判断")
        print("=" * 70)

        if inherited_ratio >= 0.999:

            print(
                "结果：预训练 Encoder 主体几乎完整继承。"
            )

            print(
                "少量 bias 参数的命名差异通常"
                "不会意味着整个 Encoder 加载失败。"
            )

        elif inherited_ratio >= 0.95:

            print(
                "结果：绝大多数预训练 Encoder "
                "参数已经成功继承。"
            )

            print(
                "存在少量兼容差异，"
                "建议重点检查未匹配参数是否仅为 bias。"
            )

        elif inherited_ratio >= 0.90:

            print(
                "结果：大部分 Encoder 参数成功继承，"
                "但兼容差异已经比较明显。"
            )

            print(
                "建议进一步检查 Transformers "
                "版本与 checkpoint 参数结构。"
            )

        else:

            print(
                "警告：Encoder预训练参数继承比例偏低。"
            )

            print(
                "不建议直接开始正式训练，"
                "应先处理模型版本兼容问题。"
            )

        print("=" * 70)


    # ========================================================
    # 释放额外CPU内存
    # ========================================================

    del checkpoint

    gc.collect()


# ============================================================
# 16. 打印第一个 Encoder Weight 的统计信息
# ============================================================

def print_first_encoder_weight(
    model: VideoMAEForVideoClassification,
) -> None:

    print("\n" + "=" * 70)
    print("Encoder 第一层权重示例")
    print("=" * 70)

    found = False

    for name, parameter in (
        model.named_parameters()
    ):

        if (
            "videomae.encoder.layer.0"
            in name
            and
            name.endswith(
                ".weight"
            )
        ):

            tensor = (
                parameter
                .detach()
                .float()
                .cpu()
            )

            print(
                f"参数：{name}"
            )

            print(
                "Shape："
                f"{tuple(tensor.shape)}"
            )

            print(
                "Mean："
                f"{tensor.mean().item():.8f}"
            )

            print(
                "Std："
                f"{tensor.std().item():.8f}"
            )

            print(
                "Min："
                f"{tensor.min().item():.8f}"
            )

            print(
                "Max："
                f"{tensor.max().item():.8f}"
            )

            found = True

            break

    if not found:

        print(
            "没有找到Encoder第一层weight。"
        )

    print("=" * 70)


# ============================================================
# 17. 计算允许的帧数差异
# ============================================================

def get_allowed_frame_difference(
    expected_frame_count: int,
) -> int:

    if expected_frame_count <= 0:

        return (
            MAX_ALLOWED_FRAME_DIFFERENCE
        )

    ratio_difference = int(
        np.ceil(
            expected_frame_count
            * FRAME_COUNT_TOLERANCE_RATIO
        )
    )

    return max(
        MAX_ALLOWED_FRAME_DIFFERENCE,
        ratio_difference,
    )


# ============================================================
# 18. 完整检查一个视频
# ============================================================

def check_video_decodable(
    video_path: Path,
) -> tuple[bool, str]:

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():

        cap.release()

        return (
            False,
            "无法打开视频",
        )

    expected_frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    decoded_frame_count = 0

    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            break

        if frame is None:
            break

        if frame.size == 0:
            break

        decoded_frame_count += 1

    cap.release()


    if decoded_frame_count == 0:

        return (
            False,
            "没有成功解码任何帧",
        )


    if decoded_frame_count < 8:

        return (
            False,
            "实际只能解码 "
            f"{decoded_frame_count} 帧，"
            "帧数过少",
        )


    if fps <= 0:

        return (
            False,
            f"FPS异常：{fps}",
        )


    if expected_frame_count > 0:

        difference = (
            expected_frame_count
            - decoded_frame_count
        )

        allowed_difference = (
            get_allowed_frame_difference(
                expected_frame_count
            )
        )

        if (
            difference
            > allowed_difference
        ):

            decode_ratio = (
                decoded_frame_count
                / expected_frame_count
            )

            return (
                False,
                "视频没有完整解码："
                f"声明 {expected_frame_count} 帧，"
                f"实际读取 {decoded_frame_count} 帧，"
                f"解码比例 {decode_ratio:.2%}",
            )

    return (
        True,
        f"{decoded_frame_count} 帧，"
        f"{fps:.2f} FPS，"
        f"{width}×{height}",
    )


# ============================================================
# 19. 读取完整视频
# ============================================================

def read_all_frames(
    video_path: Path,
    max_retries: int = MAX_VIDEO_READ_RETRIES,
) -> list[np.ndarray]:

    last_error = (
        "未知读取错误"
    )

    for attempt in range(
        1,
        max_retries + 1,
    ):

        cap = cv2.VideoCapture(
            str(
                video_path
            )
        )

        if not cap.isOpened():

            last_error = (
                "无法打开视频"
            )

            cap.release()

            print(
                "\n[视频读取警告]"
            )

            print(
                f"尝试："
                f"{attempt}/{max_retries}"
            )

            print(
                f"文件："
                f"{video_path}"
            )

            time.sleep(
                VIDEO_RETRY_WAIT_SECONDS
            )

            continue


        expected_frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        frames: list[
            np.ndarray
        ] = []


        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                break

            if frame is None:
                break

            if frame.size == 0:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(
                frame
            )


        cap.release()


        decoded_frame_count = len(
            frames
        )


        if decoded_frame_count == 0:

            last_error = (
                "整个视频没有读取到有效帧"
            )

            print(
                "\n[视频读取警告]"
            )

            print(
                f"尝试："
                f"{attempt}/{max_retries}"
            )

            print(
                f"视频："
                f"{video_path}"
            )

            time.sleep(
                VIDEO_RETRY_WAIT_SECONDS
            )

            continue


        if decoded_frame_count < 8:

            last_error = (
                "实际只能解码 "
                f"{decoded_frame_count} 帧"
            )

            time.sleep(
                VIDEO_RETRY_WAIT_SECONDS
            )

            continue


        if expected_frame_count > 0:

            difference = (
                expected_frame_count
                - decoded_frame_count
            )

            allowed_difference = (
                get_allowed_frame_difference(
                    expected_frame_count
                )
            )

            if (
                difference
                > allowed_difference
            ):

                decode_ratio = (
                    decoded_frame_count
                    / expected_frame_count
                )

                last_error = (
                    "视频未完整解码："
                    f"声明 {expected_frame_count} 帧，"
                    f"实际读取 {decoded_frame_count} 帧，"
                    f"比例 {decode_ratio:.2%}"
                )

                print(
                    "\n[视频读取警告]"
                )

                print(
                    last_error
                )

                print(
                    f"文件："
                    f"{video_path}"
                )

                time.sleep(
                    VIDEO_RETRY_WAIT_SECONDS
                )

                continue


        if attempt > 1:

            print(
                "\n[视频读取恢复]"
            )

            print(
                f"第 {attempt} 次尝试成功："
                f"{video_path}"
            )

        return frames


    raise RuntimeError(
        "\n视频连续多次读取失败。\n"
        f"视频：{video_path}\n"
        f"重试次数：{max_retries}\n"
        f"最后错误：{last_error}"
    )


# ============================================================
# 20. TRAIN：随机时间分段采样
# ============================================================

def random_segment_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
) -> list[np.ndarray]:

    total_frames = len(
        frames
    )

    if total_frames <= 0:

        raise RuntimeError(
            "视频没有有效帧。"
        )


    if total_frames < num_frames:

        indices = np.linspace(
            0,
            total_frames - 1,
            num=num_frames,
        ).round().astype(int)

        return [
            frames[index]
            for index in indices
        ]


    boundaries = np.linspace(
        0,
        total_frames,
        num=num_frames + 1,
    )


    sampled_indices: list[
        int
    ] = []


    for index in range(
        num_frames
    ):

        start = int(
            np.floor(
                boundaries[
                    index
                ]
            )
        )

        end = int(
            np.floor(
                boundaries[
                    index + 1
                ]
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

        selected_index = random.randint(
            start,
            end - 1,
        )

        sampled_indices.append(
            selected_index
        )


    return [
        frames[index]
        for index
        in sampled_indices
    ]


# ============================================================
# 21. VAL / TEST：固定中心采样
# ============================================================

def center_segment_sample_frames(
    frames: list[np.ndarray],
    num_frames: int,
) -> list[np.ndarray]:

    total_frames = len(
        frames
    )

    if total_frames <= 0:

        raise RuntimeError(
            "视频没有有效帧。"
        )


    if total_frames < num_frames:

        indices = np.linspace(
            0,
            total_frames - 1,
            num=num_frames,
        ).round().astype(int)

        return [
            frames[index]
            for index in indices
        ]


    boundaries = np.linspace(
        0,
        total_frames,
        num=num_frames + 1,
    )


    sampled_indices: list[
        int
    ] = []


    for index in range(
        num_frames
    ):

        start = int(
            np.floor(
                boundaries[
                    index
                ]
            )
        )

        end = int(
            np.floor(
                boundaries[
                    index + 1
                ]
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

        selected_index = (
            start
            + end
            - 1
        ) // 2

        sampled_indices.append(
            selected_index
        )


    return [
        frames[index]
        for index
        in sampled_indices
    ]


# ============================================================
# 22. Dataset
# ============================================================

class BasketballVideoDataset(
    Dataset
):

    def __init__(
        self,
        root_dir: Path,
        processor: VideoMAEImageProcessor,
        num_frames: int,
        training: bool,
    ) -> None:

        self.root_dir = (
            root_dir
        )

        self.processor = (
            processor
        )

        self.num_frames = (
            num_frames
        )

        self.training = (
            training
        )

        self.samples: list[
            tuple[Path, int]
        ] = []


        for class_name in CLASS_NAMES:

            class_dir = (
                root_dir
                / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    "缺少类别文件夹："
                    f"{class_dir}"
                )


            label_id = (
                LABEL_TO_ID[
                    class_name
                ]
            )


            for video_path in sorted(
                class_dir.rglob("*")
            ):

                if (
                    video_path.is_file()
                    and
                    video_path.suffix.lower()
                    in VIDEO_EXTENSIONS
                ):

                    self.samples.append(
                        (
                            video_path,
                            label_id,
                        )
                    )


        if not self.samples:

            raise RuntimeError(
                "数据集中没有找到视频："
                f"{root_dir}"
            )


        print(
            f"{root_dir.name}："
            f"{len(self.samples)} 个视频"
        )


    def __len__(
        self,
    ) -> int:

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        video_path, label = (
            self.samples[
                index
            ]
        )


        frames = read_all_frames(
            video_path
        )


        # ====================================================
        # 时间采样
        # ====================================================

        if self.training:

            frames = (
                random_segment_sample_frames(
                    frames,
                    self.num_frames,
                )
            )

        else:

            frames = (
                center_segment_sample_frames(
                    frames,
                    self.num_frames,
                )
            )


        # ====================================================
        # 水平翻转增强
        # ====================================================

        if (
            self.training
            and
            random.random() < 0.5
        ):

            frames = [

                np.ascontiguousarray(
                    frame[
                        :,
                        ::-1
                    ]
                )

                for frame
                in frames
            ]


        # ====================================================
        # VideoMAE Processor
        # ====================================================

        processed = self.processor(
            frames,
            return_tensors="pt",
        )


        pixel_values = (
            processed[
                "pixel_values"
            ]
            .squeeze(0)
        )


        return {

            "pixel_values":
                pixel_values,

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),

            "video_path":
                str(
                    video_path
                ),
        }


# ============================================================
# 23. 打印数据分布
# ============================================================

def print_dataset_distribution(
    dataset: BasketballVideoDataset,
    split_name: str,
) -> None:

    labels = [
        label
        for _, label
        in dataset.samples
    ]

    print("\n" + "-" * 70)

    print(
        f"{split_name.upper()} 类别分布"
    )

    print("-" * 70)


    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        count = labels.count(
            class_id
        )

        ratio = (
            count
            / len(
                labels
            )
        )


        print(
            f"{class_name:<24}"
            f"{count:>5} "
            f"({ratio:.2%})"
        )


# ============================================================
# 24. 训练前视频完整检查
# ============================================================

def precheck_all_videos(
    datasets: dict[
        str,
        BasketballVideoDataset
    ],
) -> None:

    print("\n" + "=" * 70)
    print("训练前完整视频解码检查")
    print("=" * 70)


    failed_records: list[
        dict[str, str]
    ] = []


    total_videos = sum(
        len(
            dataset.samples
        )
        for dataset
        in datasets.values()
    )


    current_index = 0


    for split_name, dataset in (
        datasets.items()
    ):

        print(
            "\n正在检查 "
            f"{split_name.upper()}："
            f"{len(dataset.samples)} 个视频"
        )


        for video_path, _ in (
            dataset.samples
        ):

            current_index += 1


            ok, info = (
                check_video_decodable(
                    video_path
                )
            )


            if not ok:

                print(
                    "\n"
                    f"[{current_index}/{total_videos}] "
                    "[FAILED]"
                )

                print(
                    f"视频："
                    f"{video_path}"
                )

                print(
                    f"原因："
                    f"{info}"
                )


                failed_records.append(
                    {
                        "split":
                            split_name,

                        "video_path":
                            str(
                                video_path
                            ),

                        "reason":
                            info,
                    }
                )


            elif (
                current_index == 1
                or
                current_index
                % PRECHECK_PRINT_EVERY
                == 0
                or
                current_index
                == total_videos
            ):

                print(
                    f"[{current_index}/{total_videos}] "
                    "检查正常："
                    f"{video_path.name}"
                )


    print("\n" + "=" * 70)

    print(
        f"视频总数："
        f"{total_videos}"
    )

    print(
        f"异常视频："
        f"{len(failed_records)}"
    )


    if failed_records:

        report_path = (
            OUTPUT_DIR
            / "video_precheck_failures.csv"
        )


        pd.DataFrame(
            failed_records
        ).to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig",
        )


        print(
            "\n异常视频报告："
            f"{report_path}"
        )


        raise RuntimeError(
            "\n数据集中存在无法完整解码的视频。\n"
            "请处理后重新训练。"
        )


    print(
        "所有视频均通过完整解码检查。"
    )

    print("=" * 70)


# ============================================================
# 25. Collate
# ============================================================

def collate_fn(
    examples: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    torch.Tensor
]:

    pixel_values = torch.stack(
        [
            example[
                "pixel_values"
            ]
            for example
            in examples
        ]
    )


    labels = torch.stack(
        [
            example[
                "labels"
            ]
            for example
            in examples
        ]
    )


    return {

        "pixel_values":
            pixel_values,

        "labels":
            labels,
    }


# ============================================================
# 26. 评价指标
# ============================================================

def compute_metrics(
    eval_prediction: Any,
) -> dict[str, float]:

    logits = (
        eval_prediction.predictions
    )


    if isinstance(
        logits,
        tuple,
    ):

        logits = (
            logits[
                0
            ]
        )


    predictions = np.argmax(
        logits,
        axis=1,
    )


    labels = (
        eval_prediction.label_ids
    )


    accuracy = accuracy_score(
        labels,
        predictions,
    )


    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = (
        precision_recall_fscore_support(
            labels,
            predictions,

            labels=list(
                range(
                    len(
                        CLASS_NAMES
                    )
                )
            ),

            average="macro",

            zero_division=0,
        )
    )


    (
        class_precision,
        class_recall,
        class_f1,
        class_support,
    ) = (
        precision_recall_fscore_support(
            labels,
            predictions,

            labels=list(
                range(
                    len(
                        CLASS_NAMES
                    )
                )
            ),

            average=None,

            zero_division=0,
        )
    )


    metrics: dict[
        str,
        float
    ] = {

        "accuracy":
            float(
                accuracy
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),
    }


    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        metrics[
            f"{class_name}_precision"
        ] = float(
            class_precision[
                class_id
            ]
        )

        metrics[
            f"{class_name}_recall"
        ] = float(
            class_recall[
                class_id
            ]
        )

        metrics[
            f"{class_name}_f1"
        ] = float(
            class_f1[
                class_id
            ]
        )

        metrics[
            f"{class_name}_support"
        ] = float(
            class_support[
                class_id
            ]
        )


    return metrics


# ============================================================
# 27. 保存训练日志
# ============================================================

def save_training_log(
    trainer: Trainer,
) -> None:

    if not trainer.state.log_history:

        return


    dataframe = pd.DataFrame(
        trainer.state.log_history
    )


    path = (
        OUTPUT_DIR
        / "training_log.csv"
    )


    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        f"训练日志："
        f"{path}"
    )


# ============================================================
# 28. 保存训练曲线
# ============================================================

def save_training_curves(
    trainer: Trainer,
) -> None:

    logs = (
        trainer.state.log_history
    )


    if not logs:

        return


    # ========================================================
    # Train Loss
    # ========================================================

    train_logs = [
        item
        for item
        in logs
        if (
            "loss" in item
            and
            "eval_loss"
            not in item
            and
            "step"
            in item
        )
    ]


    if train_logs:

        steps = [
            item[
                "step"
            ]
            for item
            in train_logs
        ]

        values = [
            item[
                "loss"
            ]
            for item
            in train_logs
        ]


        plt.figure(
            figsize=(
                8,
                5,
            )
        )

        plt.plot(
            steps,
            values,
        )

        plt.xlabel(
            "Training Step"
        )

        plt.ylabel(
            "Train Loss"
        )

        plt.title(
            "VideoMAE Training Loss"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / "train_loss_curve.png",
            dpi=200,
        )

        plt.close()


    # ========================================================
    # Validation Macro-F1
    # ========================================================

    f1_logs = [
        item
        for item
        in logs
        if (
            "eval_macro_f1"
            in item
            and
            "epoch"
            in item
        )
    ]


    if f1_logs:

        epochs = [
            item[
                "epoch"
            ]
            for item
            in f1_logs
        ]

        values = [
            item[
                "eval_macro_f1"
            ]
            for item
            in f1_logs
        ]


        plt.figure(
            figsize=(
                8,
                5,
            )
        )

        plt.plot(
            epochs,
            values,
            marker="o",
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Validation Macro-F1"
        )

        plt.title(
            "Validation Macro-F1"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / "val_macro_f1_curve.png",
            dpi=200,
        )

        plt.close()


    # ========================================================
    # Validation Accuracy
    # ========================================================

    accuracy_logs = [
        item
        for item
        in logs
        if (
            "eval_accuracy"
            in item
            and
            "epoch"
            in item
        )
    ]


    if accuracy_logs:

        epochs = [
            item[
                "epoch"
            ]
            for item
            in accuracy_logs
        ]

        values = [
            item[
                "eval_accuracy"
            ]
            for item
            in accuracy_logs
        ]


        plt.figure(
            figsize=(
                8,
                5,
            )
        )

        plt.plot(
            epochs,
            values,
            marker="o",
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Validation Accuracy"
        )

        plt.title(
            "Validation Accuracy"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / "val_accuracy_curve.png",
            dpi=200,
        )

        plt.close()


# ============================================================
# 29. 保存混淆矩阵
# ============================================================

def save_confusion_matrices(
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
) -> None:

    matrix = confusion_matrix(
        true_ids,
        predicted_ids,

        labels=list(
            range(
                len(
                    CLASS_NAMES
                )
            )
        ),
    )


    figure, axis = plt.subplots(
        figsize=(
            8,
            7,
        )
    )


    display = (
        ConfusionMatrixDisplay(

            confusion_matrix=(
                matrix
            ),

            display_labels=(
                CLASS_NAMES
            ),
        )
    )


    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
    )


    plt.title(
        "Basketball Shot Type Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "confusion_matrix.png",
        dpi=200,
    )

    plt.close(
        figure
    )


    # ========================================================
    # 归一化矩阵
    # ========================================================

    normalized_matrix = confusion_matrix(
        true_ids,
        predicted_ids,

        labels=list(
            range(
                len(
                    CLASS_NAMES
                )
            )
        ),

        normalize="true",
    )


    figure, axis = plt.subplots(
        figsize=(
            8,
            7,
        )
    )


    display = (
        ConfusionMatrixDisplay(

            confusion_matrix=(
                normalized_matrix
            ),

            display_labels=(
                CLASS_NAMES
            ),
        )
    )


    display.plot(
        ax=axis,
        cmap="Blues",
        values_format=".2f",
    )


    plt.title(
        "Normalized Basketball Shot Type Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "confusion_matrix_normalized.png",
        dpi=200,
    )

    plt.close(
        figure
    )


# ============================================================
# 30. 主程序
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Transformers / CUDA 环境
    # ========================================================

    print_transformers_version()


    print("\n" + "=" * 70)
    print("设备检查")
    print("=" * 70)


    print(
        f"CUDA可用："
        f"{torch.cuda.is_available()}"
    )


    if torch.cuda.is_available():

        print(
            "显卡："
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "显存："
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    else:

        print(
            "警告：当前未使用CUDA。"
        )


    # ========================================================
    # 检查模型目录
    # ========================================================

    if not MODEL_NAME.exists():

        raise FileNotFoundError(
            "模型目录不存在："
            f"{MODEL_NAME}"
        )


    # ========================================================
    # 加载 Processor
    # ========================================================

    print("\n" + "=" * 70)
    print("加载 VideoMAE ImageProcessor")
    print("=" * 70)


    processor = (
        VideoMAEImageProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )


    # ========================================================
    # 加载分类模型
    # ========================================================

    print(
        "加载 VideoMAE 三分类模型"
    )


    model = (
        VideoMAEForVideoClassification
        .from_pretrained(

            MODEL_NAME,

            num_labels=len(
                CLASS_NAMES
            ),

            label2id=(
                LABEL_TO_ID
            ),

            id2label=(
                ID_TO_LABEL
            ),

            ignore_mismatched_sizes=True,
        )
    )


    # ========================================================
    # Config检查
    # ========================================================

    print_videomae_config(
        model
    )


    # ========================================================
    # 模型参数统计
    # ========================================================

    print_model_parameter_statistics(
        model
    )


    # ========================================================
    # Encoder第一层权重示例
    # ========================================================

    print_first_encoder_weight(
        model
    )


    # ========================================================
    # 最重要：
    # 检查预训练Encoder是否真正加载
    # ========================================================

    if CHECK_PRETRAINED_ENCODER:

        check_encoder_loading(
            MODEL_NAME,
            model,
        )


    # ========================================================
    # 模型输入帧数
    # ========================================================

    num_frames = int(
        model.config.num_frames
    )


    print(
        "\n模型实际输入帧数："
        f"{num_frames}"
    )


    # ========================================================
    # Gradient Checkpointing
    # ========================================================

    model.gradient_checkpointing_enable()


    # ========================================================
    # Dataset
    # ========================================================

    print("\n" + "=" * 70)
    print("建立数据集")
    print("=" * 70)


    train_dataset = (
        BasketballVideoDataset(
            DATA_ROOT
            / "train",

            processor,

            num_frames,

            training=True,
        )
    )


    val_dataset = (
        BasketballVideoDataset(
            DATA_ROOT
            / "val",

            processor,

            num_frames,

            training=False,
        )
    )


    test_dataset = (
        BasketballVideoDataset(
            DATA_ROOT
            / "test",

            processor,

            num_frames,

            training=False,
        )
    )


    print_dataset_distribution(
        train_dataset,
        "train",
    )

    print_dataset_distribution(
        val_dataset,
        "val",
    )

    print_dataset_distribution(
        test_dataset,
        "test",
    )


    # ========================================================
    # 视频完整性检查
    # ========================================================

    if PRECHECK_ALL_VIDEOS:

        precheck_all_videos(
            {
                "train":
                    train_dataset,

                "val":
                    val_dataset,

                "test":
                    test_dataset,
            }
        )


    # ========================================================
    # 计算 Warmup Steps
    # ========================================================

    batches_per_epoch = math.ceil(
        len(
            train_dataset
        )
        /
        TRAIN_BATCH_SIZE
    )


    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        /
        GRADIENT_ACCUMULATION_STEPS
    )


    max_training_steps = (
        optimizer_steps_per_epoch
        *
        NUM_EPOCHS
    )


    warmup_steps = max(
        1,

        int(
            round(
                max_training_steps
                *
                WARMUP_RATIO
            )
        ),
    )


    print("\n" + "=" * 70)
    print("训练步数与 Warmup 计算")
    print("=" * 70)


    print(
        "训练集视频数量："
        f"{len(train_dataset)}"
    )

    print(
        "每个Epoch Batch数："
        f"{batches_per_epoch}"
    )

    print(
        "每个Epoch优化步数："
        f"{optimizer_steps_per_epoch}"
    )

    print(
        "最大训练优化步数："
        f"{max_training_steps}"
    )

    print(
        "Warmup比例："
        f"{WARMUP_RATIO:.0%}"
    )

    print(
        "Warmup Steps："
        f"{warmup_steps}"
    )

    print("=" * 70)


    # ========================================================
    # TrainingArguments
    # ========================================================

    training_args = (
        TrainingArguments(

            output_dir=str(
                OUTPUT_DIR
            ),

            num_train_epochs=(
                NUM_EPOCHS
            ),

            learning_rate=(
                LEARNING_RATE
            ),

            weight_decay=(
                WEIGHT_DECAY
            ),

            label_smoothing_factor=(
                LABEL_SMOOTHING_FACTOR
            ),

            per_device_train_batch_size=(
                TRAIN_BATCH_SIZE
            ),

            per_device_eval_batch_size=(
                EVAL_BATCH_SIZE
            ),

            gradient_accumulation_steps=(
                GRADIENT_ACCUMULATION_STEPS
            ),

            eval_strategy="epoch",

            save_strategy="epoch",

            logging_strategy="steps",

            logging_steps=10,

            load_best_model_at_end=True,

            metric_for_best_model=(
                "macro_f1"
            ),

            greater_is_better=True,

            save_total_limit=3,

            # -----------------------------------------------
            # 已改成 warmup_steps
            # 不再使用 warmup_ratio
            # -----------------------------------------------

            warmup_steps=(
                warmup_steps
            ),

            lr_scheduler_type="cosine",

            optim="adamw_torch",

            fp16=(
                torch.cuda.is_available()
            ),

            gradient_checkpointing=True,

            remove_unused_columns=False,

            dataloader_num_workers=0,

            dataloader_pin_memory=(
                torch.cuda.is_available()
            ),

            report_to="none",

            seed=(
                RANDOM_SEED
            ),

            data_seed=(
                RANDOM_SEED
            ),
        )
    )


    # ========================================================
    # EarlyStopping Callback
    # ========================================================

    callbacks = []


    if ENABLE_EARLY_STOPPING:

        callbacks.append(

            EarlyStoppingCallback(

                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                ),

                early_stopping_threshold=(
                    EARLY_STOPPING_THRESHOLD
                ),
            )
        )


    # ========================================================
    # Trainer
    # ========================================================

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=(
            train_dataset
        ),

        eval_dataset=(
            val_dataset
        ),

        data_collator=(
            collate_fn
        ),

        compute_metrics=(
            compute_metrics
        ),

        callbacks=(
            callbacks
        ),
    )


    # ========================================================
    # 输出训练配置
    # ========================================================

    print("\n" + "=" * 70)
    print("训练配置")
    print("=" * 70)


    print(
        f"最大Epoch："
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning Rate："
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight Decay："
        f"{WEIGHT_DECAY}"
    )

    print(
        "有效Batch Size："
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Warmup Steps："
        f"{warmup_steps}"
    )

    print(
        f"EarlyStopping："
        f"{ENABLE_EARLY_STOPPING}"
    )

    print(
        "EarlyStopping Patience："
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        "最佳模型指标："
        "Validation Macro-F1"
    )

    print("=" * 70)


    # ========================================================
    # 查找旧 checkpoint
    # ========================================================

    last_checkpoint = (
        get_last_checkpoint(
            str(
                OUTPUT_DIR
            )
        )
    )


    # ========================================================
    # 开始训练
    # ========================================================

    if RESUME_FROM_CHECKPOINT:

        if last_checkpoint is None:

            raise RuntimeError(
                "设置了继续训练，"
                "但没有发现checkpoint。"
            )


        print(
            "\n继续训练："
            f"{last_checkpoint}"
        )


        train_result = (
            trainer.train(
                resume_from_checkpoint=(
                    last_checkpoint
                )
            )
        )


    else:

        if last_checkpoint is not None:

            print(
                "\n检测到旧checkpoint："
            )

            print(
                last_checkpoint
            )

            print(
                "本次不恢复，"
                "从预训练VideoMAE重新开始微调。"
            )


        print("\n" + "=" * 70)
        print("开始训练")
        print("=" * 70)


        train_result = (
            trainer.train()
        )


    # ========================================================
    # 最佳模型信息
    # ========================================================

    print("\n" + "=" * 70)
    print("最佳模型信息")
    print("=" * 70)


    print(
        "实际训练结束Epoch："
        f"{trainer.state.epoch}"
    )


    print(
        f"最佳 checkpoint："
        f"{trainer.state.best_model_checkpoint}"
    )


    print(
        f"最佳验证 Macro-F1："
        f"{trainer.state.best_metric}"
    )


    # ========================================================
    # 保存训练日志与曲线
    # ========================================================

    save_training_log(
        trainer
    )

    save_training_curves(
        trainer
    )


    # ========================================================
    # 保存 best_model
    # ========================================================

    best_model_dir = (
        OUTPUT_DIR
        / "best_model"
    )


    if (
        REMOVE_OLD_BEST_MODEL_BEFORE_SAVE
        and
        best_model_dir.exists()
    ):

        shutil.rmtree(
            best_model_dir
        )


    trainer.save_model(
        str(
            best_model_dir
        )
    )


    processor.save_pretrained(
        str(
            best_model_dir
        )
    )


    # ========================================================
    # 保存训练指标
    # ========================================================

    with open(

        OUTPUT_DIR
        / "train_metrics.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(
            train_result.metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # 保存训练摘要
    # ========================================================

    training_summary = {

        "transformers_version":
            transformers.__version__,

        "max_epochs":
            NUM_EPOCHS,

        "actual_final_epoch":
            trainer.state.epoch,

        "best_model_checkpoint":
            trainer.state.best_model_checkpoint,

        "best_validation_macro_f1":
            trainer.state.best_metric,

        "early_stopping_enabled":
            ENABLE_EARLY_STOPPING,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_steps":
            warmup_steps,

        "max_training_steps":
            max_training_steps,

        "train_batch_size":
            TRAIN_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            TRAIN_BATCH_SIZE
            *
            GRADIENT_ACCUMULATION_STEPS,

        "random_seed":
            RANDOM_SEED,
    }


    with open(

        OUTPUT_DIR
        / "training_summary.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(
            training_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("开始测试集评价")
    print("=" * 70)


    prediction_output = (
        trainer.predict(
            test_dataset,
            metric_key_prefix="test",
        )
    )


    test_metrics = (
        prediction_output.metrics
    )


    # ========================================================
    # 保存测试指标
    # ========================================================

    with open(

        OUTPUT_DIR
        / "test_metrics.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(
            test_metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # 获取预测
    # ========================================================

    logits = (
        prediction_output.predictions
    )


    if isinstance(
        logits,
        tuple,
    ):

        logits = (
            logits[
                0
            ]
        )


    true_ids = np.asarray(
        prediction_output.label_ids
    )


    predicted_ids = np.argmax(
        logits,
        axis=1,
    )


    # ========================================================
    # Softmax概率
    # ========================================================

    probabilities = (
        torch.softmax(

            torch.tensor(
                logits,
                dtype=torch.float32,
            ),

            dim=1,
        )
        .cpu()
        .numpy()
    )


    # ========================================================
    # 测试集逐视频预测
    # ========================================================

    prediction_records: list[
        dict[str, Any]
    ] = []


    for sample_index, (
        sample,
        true_id,
        predicted_id,
    ) in enumerate(

        zip(
            test_dataset.samples,
            true_ids,
            predicted_ids,
        )
    ):

        video_path, _ = (
            sample
        )


        sample_probabilities = (
            probabilities[
                sample_index
            ]
        )


        sorted_indices = (
            np.argsort(
                sample_probabilities
            )[::-1]
        )


        best_probability = float(
            sample_probabilities[
                sorted_indices[
                    0
                ]
            ]
        )


        second_probability = float(
            sample_probabilities[
                sorted_indices[
                    1
                ]
            ]
        )


        margin = (
            best_probability
            -
            second_probability
        )


        record: dict[
            str,
            Any
        ] = {

            "video_path":
                str(
                    video_path
                ),

            "video_name":
                video_path.name,

            "true_label":
                ID_TO_LABEL[
                    int(
                        true_id
                    )
                ],

            "predicted_label":
                ID_TO_LABEL[
                    int(
                        predicted_id
                    )
                ],

            "correct":
                bool(
                    true_id
                    ==
                    predicted_id
                ),

            "confidence":
                best_probability,

            "top1_top2_margin":
                margin,
        }


        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            record[
                f"prob_{class_name}"
            ] = float(
                sample_probabilities[
                    class_id
                ]
            )


        prediction_records.append(
            record
        )


    prediction_csv_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )


    pd.DataFrame(
        prediction_records
    ).to_csv(
        prediction_csv_path,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Classification Report
    # ========================================================

    report = classification_report(

        true_ids,

        predicted_ids,

        labels=list(
            range(
                len(
                    CLASS_NAMES
                )
            )
        ),

        target_names=(
            CLASS_NAMES
        ),

        output_dict=True,

        zero_division=0,
    )


    report_dataframe = (
        pd.DataFrame(
            report
        )
        .transpose()
    )


    classification_report_path = (
        OUTPUT_DIR
        / "test_classification_report.csv"
    )


    report_dataframe.to_csv(
        classification_report_path,
        encoding="utf-8-sig",
    )


    with open(

        OUTPUT_DIR
        / "test_classification_report.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # 混淆矩阵
    # ========================================================

    save_confusion_matrices(
        true_ids,
        predicted_ids,
    )


    # ========================================================
    # 最终结果
    # ========================================================

    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)


    print(
        f"最佳模型："
        f"{best_model_dir}"
    )


    print(
        f"最佳 checkpoint："
        f"{trainer.state.best_model_checkpoint}"
    )


    print(
        f"最佳验证 Macro-F1："
        f"{trainer.state.best_metric}"
    )


    print(
        f"实际训练结束 Epoch："
        f"{trainer.state.epoch}"
    )


    print(
        f"测试预测结果："
        f"{prediction_csv_path}"
    )


    print(
        "Encoder加载报告："
        f"{OUTPUT_DIR / 'encoder_loading_report.csv'}"
    )


    # ========================================================
    # 总体指标
    # ========================================================

    print("\n" + "-" * 70)
    print("测试集总体指标")
    print("-" * 70)


    important_metrics = [

        "test_loss",

        "test_accuracy",

        "test_macro_precision",

        "test_macro_recall",

        "test_macro_f1",
    ]


    for key in important_metrics:

        if key in test_metrics:

            print(
                f"{key}: "
                f"{test_metrics[key]}"
            )


    # ========================================================
    # 每类别指标
    # ========================================================

    print("\n" + "-" * 70)
    print("各类别测试结果")
    print("-" * 70)


    for class_name in CLASS_NAMES:

        precision_key = (
            f"test_{class_name}_precision"
        )

        recall_key = (
            f"test_{class_name}_recall"
        )

        f1_key = (
            f"test_{class_name}_f1"
        )

        support_key = (
            f"test_{class_name}_support"
        )


        print(
            "\n"
            f"{CHINESE_LABELS[class_name]} "
            f"({class_name})"
        )


        if precision_key in test_metrics:

            print(
                "  Precision："
                f"{test_metrics[precision_key]:.4f}"
            )


        if recall_key in test_metrics:

            print(
                "  Recall："
                f"{test_metrics[recall_key]:.4f}"
            )


        if f1_key in test_metrics:

            print(
                "  F1："
                f"{test_metrics[f1_key]:.4f}"
            )


        if support_key in test_metrics:

            print(
                "  Support："
                f"{int(test_metrics[support_key])}"
            )


    print("\n" + "=" * 70)

    print(
        "全部结果保存目录："
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 70)


# ============================================================
# 31. 程序入口
# ============================================================

if __name__ == "__main__":

    main()