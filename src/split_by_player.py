from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(
    r"F:\BasketballVideoMAE"
)

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw_by_class"
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "split"
)


# ============================================================
# 2. 三个投篮类别
# ============================================================

CLASS_NAMES = [
    "one_stage",
    "one_point_five_stage",
    "two_stage",
]


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# 3. 随机种子
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 4. Train / Val / Test 球员比例
# ============================================================
#
# 原来：
#
# 70 / 15 / 15
#
# 对你当前的数据来说，
# VAL 和 TEST 中球员数量太少，
# 容易出现：
#
# one_point_five_stage
# 12个视频
# 只有1～2名球员
#
#
# 现在改为：
#
# 60 / 20 / 20
#
# 给 VAL 和 TEST 更多独立球员。
# ============================================================

TRAIN_SIZE = 0.70

VAL_SIZE = 0.15

TEST_SIZE = 0.15


# ============================================================
# 5. 最多搜索多少种划分
# ============================================================
#
# 由于现在增加了比较严格的：
#
# 每个split
# 每个类别
# 至少3名不同球员
#
# 因此提高随机搜索次数。
# ============================================================

MAX_SPLIT_ATTEMPTS = 50_000


# ============================================================
# 6. 每个 split 总体最低球员数
# ============================================================

MIN_PLAYERS_PER_SPLIT = 3


# ============================================================
# 7. 每个 split 每个类别最低视频数量
# ============================================================

MIN_SAMPLES_PER_CLASS = 1


# ============================================================
# 8. 核心：
# 每个 split 每个类别至少来自多少名不同球员
# ============================================================
#
# TRAIN：
#
# one_stage             >= 3名球员
# one_point_five_stage  >= 3名球员
# two_stage             >= 3名球员
#
#
# VAL：
#
# 三个类别也都 >= 3名
#
#
# TEST：
#
# 三个类别也都 >= 3名
#
#
# 这样就不会再出现：
#
# TEST一点五段式12个视频
# 但只来自1个球员
# ============================================================

MIN_PLAYERS_PER_CLASS_PER_SPLIT = {
    "train": 3,
    "val": 3,
    "test": 3,
}


# ============================================================
# 9. 划分评分权重
# ============================================================
#
# 满足上面的硬约束后，
# 还需要从很多合格划分中选出最好的。
#
# 下面两个参数决定：
#
# 1. 是否尽量平衡每类不同球员的数量
# 2. 是否避免一个球员贡献某类别过多视频
# ============================================================

CLASS_PLAYER_BALANCE_WEIGHT = 1.0

PLAYER_CONCENTRATION_WEIGHT = 0.5


# ============================================================
# 10. 收集所有原始视频
# ============================================================

def collect_samples() -> pd.DataFrame:
    """
    从：

        data/raw_by_class/

    中扫描三个类别的视频。

    文件名示例：

        P001_session01_s1.mp4

    第一个下划线之前：

        P001

    被作为 player_id。
    """

    records: list[
        dict[str, str]
    ] = []


    for class_name in CLASS_NAMES:

        class_dir = (
            RAW_ROOT
            / class_name
        )


        if not class_dir.exists():

            raise FileNotFoundError(
                "类别文件夹不存在："
                f"{class_dir}"
            )


        for video_path in sorted(
            class_dir.rglob("*")
        ):

            if not video_path.is_file():
                continue


            if (
                video_path.suffix.lower()
                not in VIDEO_EXTENSIONS
            ):
                continue


            # =================================================
            # 从文件名解析 player_id
            # =================================================

            parts = (
                video_path.stem
                .split("_")
            )


            if len(parts) < 2:

                raise ValueError(
                    "\n无法从视频文件名解析球员ID：\n"
                    f"{video_path}\n\n"
                    "正确示例：\n"
                    "P001_session01_s1.mp4"
                )


            player_id = (
                parts[0]
                .strip()
            )


            if not player_id:

                raise ValueError(
                    "球员ID为空："
                    f"{video_path}"
                )


            records.append(
                {
                    "path":
                        str(video_path),

                    "filename":
                        video_path.name,

                    "class_name":
                        class_name,

                    "player_id":
                        player_id,
                }
            )


    if not records:

        raise RuntimeError(
            "没有找到任何视频："
            f"{RAW_ROOT}"
        )


    dataframe = (
        pd.DataFrame(
            records
        )
    )


    # ========================================================
    # 检查完全重复路径
    # ========================================================

    duplicated_paths = (
        dataframe[
            dataframe[
                "path"
            ].duplicated(
                keep=False
            )
        ]
    )


    if not duplicated_paths.empty:

        raise RuntimeError(
            "发现完全重复的视频路径：\n"
            f"{duplicated_paths.to_string(index=False)}"
        )


    return dataframe


# ============================================================
# 11. 检查同名视频是否跨类别重复
# ============================================================

def verify_cross_class_filename_conflicts(
    dataframe: pd.DataFrame,
) -> None:
    """
    防止重新标注以后：

    P020_session01_s1.mp4

    同时残留在：

    one_point_five_stage

    和：

    two_stage

    中。

    如果出现，
    说明旧标签文件可能没有清理。
    """

    duplicated = dataframe[
        dataframe.duplicated(
            subset=[
                "filename"
            ],
            keep=False,
        )
    ].copy()


    if duplicated.empty:
        return


    conflict_groups: list[
        pd.DataFrame
    ] = []


    for _, group in (
        duplicated.groupby(
            "filename"
        )
    ):

        if (
            group[
                "class_name"
            ].nunique()
            > 1
        ):

            conflict_groups.append(
                group
            )


    if not conflict_groups:
        return


    conflict_dataframe = (
        pd.concat(
            conflict_groups,
            ignore_index=True,
        )
        .sort_values(
            [
                "filename",
                "class_name",
            ]
        )
    )


    raise RuntimeError(
        "\n发现同名视频同时存在于不同类别！\n\n"
        "这通常说明重新标注以后，"
        "旧类别中的视频没有删除。\n\n"
        f"{conflict_dataframe.to_string(index=False)}"
        "\n\n"
        "请先检查并清理：\n"
        f"{RAW_ROOT}\n"
        "然后重新运行 split_by_player.py。"
    )


# ============================================================
# 12. 构建球员 × 类别统计表
# ============================================================

def build_player_class_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    table = pd.crosstab(
        dataframe[
            "player_id"
        ],
        dataframe[
            "class_name"
        ],
    )


    return table.reindex(
        columns=CLASS_NAMES,
        fill_value=0,
    )


# ============================================================
# 13. 打印球员 × 类别分布
# ============================================================

def print_player_class_distribution(
    dataframe: pd.DataFrame,
) -> None:

    table = (
        build_player_class_table(
            dataframe
        )
    )


    print("\n" + "=" * 70)

    print(
        "当前球员—类别分布"
    )

    print("=" * 70)


    print(
        table.to_string()
    )


    # ========================================================
    # 每个球员覆盖多少类别
    # ========================================================

    class_coverage = pd.Series(
        np.count_nonzero(
            table.to_numpy() > 0,
            axis=1,
        ),
        index=table.index,
        dtype="int64",
    )


    print("\n球员覆盖类别数量：")

    print(
        class_coverage
        .value_counts()
        .sort_index()
        .to_string()
    )


    # ========================================================
    # 只提供一个类别的球员
    # ========================================================

    single_class_players = (
        class_coverage[
            class_coverage == 1
        ]
        .index
        .tolist()
    )


    if single_class_players:

        print(
            "\n警告：以下球员只提供一个类别："
        )

        print(
            ", ".join(
                single_class_players
            )
        )


    # ========================================================
    # 每一个类别有多少不同球员
    # ========================================================

    print(
        "\n每个类别的不同球员数量："
    )


    for class_name in CLASS_NAMES:

        player_count = int(
            (
                table[
                    class_name
                ]
                > 0
            )
            .sum()
        )


        print(
            f"  {class_name:<24}"
            f"{player_count} 名球员"
        )


# ============================================================
# 14. 检查每个类别是否有足够的不同球员
# ============================================================

def verify_class_player_capacity(
    dataframe: pd.DataFrame,
) -> None:
    """
    当前要求：

    TRAIN >= 3
    VAL   >= 3
    TEST  >= 3

    同一个球员又不能跨集合。

    因此理论上每个类别至少需要：

        3 + 3 + 3
        = 9

    名不同球员。
    """

    required_total = sum(
        MIN_PLAYERS_PER_CLASS_PER_SPLIT
        .values()
    )


    print("\n" + "=" * 70)

    print(
        "每个类别的球员覆盖能力检查"
    )

    print("=" * 70)


    errors: list[str] = []


    for class_name in CLASS_NAMES:

        class_dataframe = (
            dataframe.loc[
                dataframe[
                    "class_name"
                ]
                == class_name
            ]
        )


        players = sorted(
            class_dataframe[
                "player_id"
            ]
            .unique()
            .tolist()
        )


        player_count = len(
            players
        )


        print(
            f"{class_name:<24}"
            f"不同球员：{player_count:<3} "
            f"理论最低需求：{required_total}"
        )


        print(
            "  "
            + ", ".join(
                players
            )
        )


        if (
            player_count
            < required_total
        ):

            errors.append(
                f"{class_name}："
                f"当前只有 {player_count} 名球员，"
                f"至少需要 {required_total} 名。"
            )


    print("=" * 70)


    if errors:

        raise RuntimeError(
            "\n当前数据无法满足"
            "每个类别在TRAIN/VAL/TEST"
            "至少3名不同球员的要求。\n\n"
            + "\n".join(errors)
            + "\n\n"
            "建议优先增加新的不同球员，"
            "而不是继续增加原有球员的视频数量。"
        )


# ============================================================
# 15. 检查球员泄漏
# ============================================================

def verify_no_player_leakage(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    train_players = set(
        splits[
            "train"
        ][
            "player_id"
        ]
    )


    val_players = set(
        splits[
            "val"
        ][
            "player_id"
        ]
    )


    test_players = set(
        splits[
            "test"
        ][
            "player_id"
        ]
    )


    train_val_overlap = (
        train_players
        &
        val_players
    )


    train_test_overlap = (
        train_players
        &
        test_players
    )


    val_test_overlap = (
        val_players
        &
        test_players
    )


    if train_val_overlap:

        raise RuntimeError(
            "TRAIN 和 VAL 有重复球员："
            f"{sorted(train_val_overlap)}"
        )


    if train_test_overlap:

        raise RuntimeError(
            "TRAIN 和 TEST 有重复球员："
            f"{sorted(train_test_overlap)}"
        )


    if val_test_overlap:

        raise RuntimeError(
            "VAL 和 TEST 有重复球员："
            f"{sorted(val_test_overlap)}"
        )


# ============================================================
# 16. 检查三个集合是否都有全部类别
# ============================================================

def verify_all_classes(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    expected_classes = set(
        CLASS_NAMES
    )


    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        actual_classes = set(
            split_dataframe[
                "class_name"
            ]
            .unique()
        )


        missing = (
            expected_classes
            -
            actual_classes
        )


        if missing:

            raise RuntimeError(
                f"{split_name.upper()} "
                f"缺少类别："
                f"{sorted(missing)}"
            )


# ============================================================
# 17. 检查每个 split 总球员数量
# ============================================================

def verify_minimum_players(
    splits: dict[
        str,
        pd.DataFrame
    ],
    minimum: int,
) -> None:

    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        player_count = int(
            split_dataframe[
                "player_id"
            ]
            .nunique()
        )


        if (
            player_count
            < minimum
        ):

            raise RuntimeError(
                f"{split_name.upper()} "
                f"只有 {player_count} 名球员，"
                f"最低需要 {minimum} 名。"
            )


# ============================================================
# 18. 检查每个类别最低视频数量
# ============================================================

def verify_minimum_class_samples(
    splits: dict[
        str,
        pd.DataFrame
    ],
    minimum: int,
) -> None:

    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        counts = (
            split_dataframe[
                "class_name"
            ]
            .value_counts()
            .reindex(
                CLASS_NAMES,
                fill_value=0,
            )
        )


        invalid = (
            counts[
                counts < minimum
            ]
        )


        if not invalid.empty:

            raise RuntimeError(
                f"{split_name.upper()} "
                f"存在类别视频数量少于 {minimum}：\n"
                f"{invalid.to_string()}"
            )


# ============================================================
# 19. 核心：
# 检查每个split的每个类别有多少不同球员
# ============================================================

def verify_minimum_players_per_class(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    errors: list[str] = []


    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        minimum = int(
            MIN_PLAYERS_PER_CLASS_PER_SPLIT[
                split_name
            ]
        )


        for class_name in CLASS_NAMES:

            class_dataframe = (
                split_dataframe.loc[
                    split_dataframe[
                        "class_name"
                    ]
                    == class_name
                ]
            )


            players = sorted(
                class_dataframe[
                    "player_id"
                ]
                .unique()
                .tolist()
            )


            player_count = len(
                players
            )


            if (
                player_count
                < minimum
            ):

                errors.append(
                    f"{split_name.upper()} / "
                    f"{class_name}："
                    f"{player_count} 名球员 "
                    f"{players}，"
                    f"最低要求 {minimum} 名。"
                )


    if errors:

        raise RuntimeError(
            "\n每个类别的球员覆盖数量不足：\n\n"
            + "\n".join(
                errors
            )
        )


# ============================================================
# 20. 检查目标文件名是否重复
# ============================================================

def verify_unique_target_names(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        duplicated = (
            split_dataframe[
                split_dataframe.duplicated(
                    subset=[
                        "class_name",
                        "filename",
                    ],
                    keep=False,
                )
            ]
        )


        if not duplicated.empty:

            raise RuntimeError(
                f"{split_name.upper()} "
                "发现同类别同名视频：\n"
                f"{duplicated.to_string(index=False)}"
            )


# ============================================================
# 21. 快速判断候选划分是否满足硬约束
# ============================================================

def candidate_is_valid(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> bool:

    expected_classes = set(
        CLASS_NAMES
    )


    # ========================================================
    # 不能有空集合
    # ========================================================

    if any(
        split_dataframe.empty
        for split_dataframe
        in splits.values()
    ):

        return False


    # ========================================================
    # 每个 split 总体至少若干球员
    # ========================================================

    for split_dataframe in (
        splits.values()
    ):

        if (
            split_dataframe[
                "player_id"
            ]
            .nunique()
            <
            MIN_PLAYERS_PER_SPLIT
        ):

            return False


    # ========================================================
    # 每个 split 必须包含三个类别
    # ========================================================

    for split_dataframe in (
        splits.values()
    ):

        actual_classes = set(
            split_dataframe[
                "class_name"
            ]
            .unique()
        )


        if (
            actual_classes
            !=
            expected_classes
        ):

            return False


    # ========================================================
    # 每个 split 每个类别最低视频数量
    # ========================================================

    for split_dataframe in (
        splits.values()
    ):

        counts = (
            split_dataframe[
                "class_name"
            ]
            .value_counts()
            .reindex(
                CLASS_NAMES,
                fill_value=0,
            )
        )


        if (
            counts
            <
            MIN_SAMPLES_PER_CLASS
        ).any():

            return False


    # ========================================================
    # 最重要：
    #
    # 每个split
    # 每个类别
    # 至少3名不同球员
    # ========================================================

    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        required_players = int(
            MIN_PLAYERS_PER_CLASS_PER_SPLIT[
                split_name
            ]
        )


        for class_name in CLASS_NAMES:

            player_count = int(
                split_dataframe.loc[
                    split_dataframe[
                        "class_name"
                    ]
                    == class_name,
                    "player_id",
                ]
                .nunique()
            )


            if (
                player_count
                <
                required_players
            ):

                return False


    # ========================================================
    # 同一球员不能跨集合
    # ========================================================

    try:

        verify_no_player_leakage(
            splits
        )

    except RuntimeError:

        return False


    return True


# ============================================================
# 22. 计算候选划分质量分数
# ============================================================

def calculate_balance_score(
    splits: dict[
        str,
        pd.DataFrame
    ],
    total_samples: int,
) -> float:
    """
    分数越小越好。

    综合考虑：

    1. 每个split内部三类视频比例；
    2. train/val/test总视频比例；
    3. 每个类别的视频划分比例；
    4. 每个类别的不同球员划分比例；
    5. 单一球员是否占某类别过多视频。
    """

    ideal_class_ratio = (
        1.0
        /
        len(
            CLASS_NAMES
        )
    )


    target_split_ratios = {
        "train":
            TRAIN_SIZE,

        "val":
            VAL_SIZE,

        "test":
            TEST_SIZE,
    }


    score = 0.0


    # ========================================================
    # 1. 每个split内部的类别均衡性
    # ========================================================

    for split_dataframe in (
        splits.values()
    ):

        proportions = (
            split_dataframe[
                "class_name"
            ]
            .value_counts(
                normalize=True
            )
            .reindex(
                CLASS_NAMES,
                fill_value=0.0,
            )
        )


        score += float(
            (
                (
                    proportions
                    -
                    ideal_class_ratio
                )
                ** 2
            ).sum()
        )


    # ========================================================
    # 2. train/val/test总视频数量比例
    # ========================================================

    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        actual_ratio = (
            len(
                split_dataframe
            )
            /
            total_samples
        )


        target_ratio = (
            target_split_ratios[
                split_name
            ]
        )


        score += (
            0.5
            *
            (
                actual_ratio
                -
                target_ratio
            )
            ** 2
        )


    # ========================================================
    # 3. 每个类别的视频分配比例
    # ========================================================

    for class_name in CLASS_NAMES:

        class_total = sum(
            int(
                np.count_nonzero(
                    split_dataframe[
                        "class_name"
                    ].to_numpy()
                    ==
                    class_name
                )
            )
            for split_dataframe
            in splits.values()
        )


        if class_total == 0:

            return float(
                "inf"
            )


        for (
            split_name,
            split_dataframe,
        ) in splits.items():

            class_count = int(
                np.count_nonzero(
                    split_dataframe[
                        "class_name"
                    ].to_numpy()
                    ==
                    class_name
                )
            )


            actual_ratio = (
                class_count
                /
                class_total
            )


            target_ratio = (
                target_split_ratios[
                    split_name
                ]
            )


            score += (
                0.5
                *
                (
                    actual_ratio
                    -
                    target_ratio
                )
                ** 2
            )


    # ========================================================
    # 4. 每个类别的不同球员分配比例
    # ========================================================

    for class_name in CLASS_NAMES:

        class_player_counts = {
            split_name:
                int(
                    split_dataframe.loc[
                        split_dataframe[
                            "class_name"
                        ]
                        == class_name,
                        "player_id",
                    ]
                    .nunique()
                )

            for (
                split_name,
                split_dataframe,
            )
            in splits.items()
        }


        total_class_players = sum(
            class_player_counts
            .values()
        )


        if total_class_players == 0:

            return float(
                "inf"
            )


        for split_name in (
            "train",
            "val",
            "test",
        ):

            actual_ratio = (
                class_player_counts[
                    split_name
                ]
                /
                total_class_players
            )


            target_ratio = (
                target_split_ratios[
                    split_name
                ]
            )


            score += (
                CLASS_PLAYER_BALANCE_WEIGHT
                *
                (
                    actual_ratio
                    -
                    target_ratio
                )
                ** 2
            )


    # ========================================================
    # 5. 单一球员的视频集中度
    # ========================================================
    #
    # 例如：
    #
    # 12个一点五段式：
    #
    # P001 = 4
    # P002 = 4
    # P003 = 4
    #
    # 比：
    #
    # P001 = 10
    # P002 = 1
    # P003 = 1
    #
    # 更好。
    # ========================================================

    for split_dataframe in (
        splits.values()
    ):

        for class_name in CLASS_NAMES:

            class_dataframe = (
                split_dataframe.loc[
                    split_dataframe[
                        "class_name"
                    ]
                    == class_name
                ]
            )


            if class_dataframe.empty:

                return float(
                    "inf"
                )


            player_counts = (
                class_dataframe[
                    "player_id"
                ]
                .value_counts()
                .astype(float)
            )


            shares = (
                player_counts
                /
                player_counts.sum()
            )


            concentration = float(
                np.square(
                    shares.to_numpy()
                ).sum()
            )


            score += (
                PLAYER_CONCENTRATION_WEIGHT
                *
                concentration
            )


    return float(
        score
    )


# ============================================================
# 23. 搜索最佳按球员划分
# ============================================================

def split_by_group(
    dataframe: pd.DataFrame,
) -> dict[
    str,
    pd.DataFrame
]:

    player_count = int(
        dataframe[
            "player_id"
        ]
        .nunique()
    )


    # ========================================================
    # 基础球员数量检查
    # ========================================================

    minimum_total_players = (
        MIN_PLAYERS_PER_SPLIT
        *
        3
    )


    if (
        player_count
        <
        minimum_total_players
    ):

        raise RuntimeError(
            f"当前总共只有 {player_count} 名球员。\n"
            f"至少需要 {minimum_total_players} 名球员。"
        )


    print("\n" + "=" * 70)

    print(
        "目标球员划分比例"
    )

    print("=" * 70)


    print(
        f"TRAIN："
        f"{TRAIN_SIZE:.0%}"
    )


    print(
        f"VAL："
        f"{VAL_SIZE:.0%}"
    )


    print(
        f"TEST："
        f"{TEST_SIZE:.0%}"
    )


    print(
        f"预计TRAIN球员："
        f"{round(player_count * TRAIN_SIZE)}"
    )


    print(
        f"预计VAL球员："
        f"{round(player_count * VAL_SIZE)}"
    )


    print(
        f"预计TEST球员："
        f"{round(player_count * TEST_SIZE)}"
    )


    print("=" * 70)


    best_splits: (
        dict[
            str,
            pd.DataFrame
        ]
        |
        None
    ) = None


    best_score = float(
        "inf"
    )


    best_seed: (
        int
        |
        None
    ) = None


    valid_candidate_count = 0


    total_samples = len(
        dataframe
    )


    # ========================================================
    # TEMP中多少比例进入VAL
    # ========================================================
    #
    # 当前：
    #
    # 20 / (20 + 20)
    # =
    # 0.5
    # ========================================================

    val_fraction_of_temp = (
        VAL_SIZE
        /
        (
            VAL_SIZE
            +
            TEST_SIZE
        )
    )


    # ========================================================
    # 搜索大量随机种子
    # ========================================================

    for attempt in range(
        MAX_SPLIT_ATTEMPTS
    ):

        seed = (
            RANDOM_SEED
            +
            attempt
        )


        # ====================================================
        # 第一次划分：
        #
        # 约60%的球员 → TRAIN
        #
        # 剩下约40% → TEMP
        # ====================================================

        splitter_1 = GroupShuffleSplit(
            n_splits=1,
            train_size=TRAIN_SIZE,
            random_state=seed,
        )


        try:

            (
                train_indices,
                temp_indices,
            ) = next(
                splitter_1.split(
                    dataframe,
                    groups=dataframe[
                        "player_id"
                    ],
                )
            )

        except ValueError:

            continue


        train_dataframe = (
            dataframe
            .iloc[
                train_indices
            ]
            .reset_index(
                drop=True
            )
        )


        temp_dataframe = (
            dataframe
            .iloc[
                temp_indices
            ]
            .reset_index(
                drop=True
            )
        )


        temp_player_count = int(
            temp_dataframe[
                "player_id"
            ]
            .nunique()
        )


        if (
            temp_player_count
            <
            MIN_PLAYERS_PER_SPLIT
            *
            2
        ):

            continue


        # ====================================================
        # 第二次划分：
        #
        # TEMP →
        #
        # VAL
        # TEST
        #
        # 当前是50% / 50%
        #
        # 因此总体：
        #
        # TRAIN 60%
        # VAL   20%
        # TEST  20%
        # ====================================================

        splitter_2 = GroupShuffleSplit(
            n_splits=1,
            train_size=(
                val_fraction_of_temp
            ),
            random_state=(
                seed
                +
                100_000
            ),
        )


        try:

            (
                val_indices,
                test_indices,
            ) = next(
                splitter_2.split(
                    temp_dataframe,
                    groups=temp_dataframe[
                        "player_id"
                    ],
                )
            )

        except ValueError:

            continue


        val_dataframe = (
            temp_dataframe
            .iloc[
                val_indices
            ]
            .reset_index(
                drop=True
            )
        )


        test_dataframe = (
            temp_dataframe
            .iloc[
                test_indices
            ]
            .reset_index(
                drop=True
            )
        )


        candidate = {
            "train":
                train_dataframe,

            "val":
                val_dataframe,

            "test":
                test_dataframe,
        }


        # ====================================================
        # 不满足硬约束的直接淘汰
        # ====================================================

        if not candidate_is_valid(
            candidate
        ):

            continue


        valid_candidate_count += 1


        score = (
            calculate_balance_score(
                candidate,
                total_samples=(
                    total_samples
                ),
            )
        )


        # ====================================================
        # 找到更好的划分
        # ====================================================

        if (
            score
            <
            best_score
        ):

            best_score = score

            best_splits = (
                candidate
            )

            best_seed = (
                seed
            )


    # ========================================================
    # 一个满足要求的方案都没有
    # ========================================================

    if best_splits is None:

        player_class_table = (
            build_player_class_table(
                dataframe
            )
        )


        raise RuntimeError(
            f"\n尝试了 {MAX_SPLIT_ATTEMPTS} 种划分，"
            "仍没有找到满足全部条件的数据划分。\n\n"

            "当前硬约束：\n"

            "- train、val、test 球员互不重叠；\n"

            "- train、val、test 都包含三个类别；\n"

            f"- 每个集合至少 "
            f"{MIN_PLAYERS_PER_SPLIT} 名球员；\n"

            f"- 每个集合每个类别至少 "
            f"{MIN_SAMPLES_PER_CLASS} 个视频；\n"

            "- 每个集合每个类别至少来自：\n"

            f"  TRAIN："
            f"{MIN_PLAYERS_PER_CLASS_PER_SPLIT['train']} 名\n"

            f"  VAL："
            f"{MIN_PLAYERS_PER_CLASS_PER_SPLIT['val']} 名\n"

            f"  TEST："
            f"{MIN_PLAYERS_PER_CLASS_PER_SPLIT['test']} 名\n\n"

            "当前比例：\n"

            f"TRAIN = {TRAIN_SIZE:.0%}\n"
            f"VAL   = {VAL_SIZE:.0%}\n"
            f"TEST  = {TEST_SIZE:.0%}\n\n"

            "可能原因：\n"

            "1. 某个类别不同球员数量仍然不足；\n"

            "2. 类别和球员绑定过于严重；\n"

            "3. VAL/TEST中可用的多类别球员太少；\n"

            "4. 当前数据结构无法同时满足这些硬约束。\n\n"

            "当前球员—类别分布：\n"

            f"{player_class_table.to_string()}"
        )


    print("\n" + "=" * 70)

    print(
        "找到合格的数据划分"
    )

    print("=" * 70)


    print(
        f"采用随机种子："
        f"{best_seed}"
    )


    print(
        "合格候选数量："
        f"{valid_candidate_count}"
    )


    print(
        "最佳平衡分数："
        f"{best_score:.8f}"
    )


    print("=" * 70)


    return best_splits


# ============================================================
# 24. 打印总体划分统计
# ============================================================

def print_statistics(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    total_videos = sum(
        len(
            split_dataframe
        )
        for split_dataframe
        in splits.values()
    )


    all_players = set().union(
        *[
            set(
                split_dataframe[
                    "player_id"
                ]
            )
            for split_dataframe
            in splits.values()
        ]
    )


    for split_name in (
        "train",
        "val",
        "test",
    ):

        split_dataframe = (
            splits[
                split_name
            ]
        )


        video_count = len(
            split_dataframe
        )


        player_count = int(
            split_dataframe[
                "player_id"
            ]
            .nunique()
        )


        video_ratio = (
            video_count
            /
            total_videos
        )


        player_ratio = (
            player_count
            /
            len(
                all_players
            )
        )


        players = sorted(
            split_dataframe[
                "player_id"
            ]
            .unique()
            .tolist()
        )


        class_counts = (
            split_dataframe[
                "class_name"
            ]
            .value_counts()
            .reindex(
                CLASS_NAMES,
                fill_value=0,
            )
        )


        print("\n" + "=" * 70)

        print(
            f"{split_name.upper()} 数据集"
        )

        print("=" * 70)


        print(
            f"视频数量："
            f"{video_count} "
            f"({video_ratio:.2%})"
        )


        print(
            f"球员数量："
            f"{player_count} "
            f"({player_ratio:.2%})"
        )


        print(
            "球员："
            f"{', '.join(players)}"
        )


        print(
            "\n类别视频数量："
        )


        print(
            class_counts.to_string()
        )


# ============================================================
# 25. 打印每个类别的具体球员来源
# ============================================================

def print_split_class_player_distribution(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    print("\n" + "=" * 70)

    print(
        "各数据集：每个类别的球员来源"
    )

    print("=" * 70)


    for split_name in (
        "train",
        "val",
        "test",
    ):

        split_dataframe = (
            splits[
                split_name
            ]
        )


        print(
            f"\n[{split_name.upper()}]"
        )


        for class_name in CLASS_NAMES:

            class_dataframe = (
                split_dataframe.loc[
                    split_dataframe[
                        "class_name"
                    ]
                    == class_name
                ]
            )


            player_counts = (
                class_dataframe[
                    "player_id"
                ]
                .value_counts()
                .sort_index()
            )


            player_count = int(
                player_counts.size
            )


            details = ", ".join(
                f"{player_id}:{int(count)}"
                for (
                    player_id,
                    count,
                )
                in player_counts.items()
            )


            print(
                f"  {class_name:<24}"
                f"球员数={player_count:<3} "
                f"视频数={len(class_dataframe):<4} "
                f"来源=[{details}]"
            )


    print("=" * 70)


# ============================================================
# 26. 复制视频到 split
# ============================================================

def copy_split_videos(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:
    """
    每次生成新的split时：

    删除旧：

        data/split

    然后重新创建：

        train
        val
        test
    """

    if SPLIT_ROOT.exists():

        print(
            "\n删除旧的数据划分："
            f"{SPLIT_ROOT}"
        )


        shutil.rmtree(
            SPLIT_ROOT
        )


    SPLIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        for _, row in (
            split_dataframe.iterrows()
        ):

            source = Path(
                row[
                    "path"
                ]
            )


            if not source.exists():

                raise FileNotFoundError(
                    "源视频不存在："
                    f"{source}"
                )


            target_directory = (
                SPLIT_ROOT
                /
                split_name
                /
                row[
                    "class_name"
                ]
            )


            target_directory.mkdir(
                parents=True,
                exist_ok=True,
            )


            target = (
                target_directory
                /
                source.name
            )


            if target.exists():

                raise FileExistsError(
                    "目标视频已经存在：\n"
                    f"{target}"
                )


            shutil.copy2(
                source,
                target,
            )


        # ====================================================
        # 保存对应 CSV
        # ====================================================

        csv_dataframe = (
            split_dataframe.copy()
        )


        csv_dataframe[
            "split_path"
        ] = (
            csv_dataframe.apply(
                lambda row: str(
                    SPLIT_ROOT
                    /
                    split_name
                    /
                    row[
                        "class_name"
                    ]
                    /
                    row[
                        "filename"
                    ]
                ),
                axis=1,
            )
        )


        csv_dataframe.to_csv(
            SPLIT_ROOT
            /
            f"{split_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# 27. 保存 split_summary.csv
# ============================================================

def save_split_summary(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    records: list[
        dict[
            str,
            object
        ]
    ] = []


    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        for class_name in CLASS_NAMES:

            class_dataframe = (
                split_dataframe.loc[
                    split_dataframe[
                        "class_name"
                    ]
                    == class_name
                ]
            )


            player_video_counts = (
                class_dataframe[
                    "player_id"
                ]
                .value_counts()
                .sort_index()
            )


            player_count = int(
                player_video_counts.size
            )


            if player_count > 0:

                max_player_videos = int(
                    player_video_counts.max()
                )

            else:

                max_player_videos = 0


            if len(
                class_dataframe
            ) > 0:

                max_player_share = (
                    max_player_videos
                    /
                    len(
                        class_dataframe
                    )
                )

            else:

                max_player_share = 0.0


            player_details = ";".join(
                f"{player_id}:{int(count)}"
                for (
                    player_id,
                    count,
                )
                in player_video_counts.items()
            )


            records.append(
                {
                    "split":
                        split_name,

                    "class_name":
                        class_name,

                    "video_count":
                        int(
                            len(
                                class_dataframe
                            )
                        ),

                    "player_count":
                        player_count,

                    "min_required_players":
                        int(
                            MIN_PLAYERS_PER_CLASS_PER_SPLIT[
                                split_name
                            ]
                        ),

                    "max_videos_from_one_player":
                        max_player_videos,

                    "max_player_share":
                        float(
                            max_player_share
                        ),

                    "player_video_counts":
                        player_details,
                }
            )


    summary_dataframe = (
        pd.DataFrame(
            records
        )
    )


    summary_path = (
        SPLIT_ROOT
        /
        "split_summary.csv"
    )


    summary_dataframe.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        f"划分摘要："
        f"{summary_path}"
    )


# ============================================================
# 28. 保存每个球员属于哪个 split
# ============================================================

def save_player_split_assignment(
    splits: dict[
        str,
        pd.DataFrame
    ],
) -> None:

    records: list[
        dict[
            str,
            object
        ]
    ] = []


    for (
        split_name,
        split_dataframe,
    ) in splits.items():

        for player_id in sorted(
            split_dataframe[
                "player_id"
            ]
            .unique()
        ):

            player_dataframe = (
                split_dataframe.loc[
                    split_dataframe[
                        "player_id"
                    ]
                    == player_id
                ]
            )


            record: dict[
                str,
                object
            ] = {
                "split":
                    split_name,

                "player_id":
                    player_id,

                "video_count":
                    int(
                        len(
                            player_dataframe
                        )
                    ),
            }


            for class_name in CLASS_NAMES:

                record[
                    f"{class_name}_count"
                ] = int(
                    np.count_nonzero(
                        player_dataframe[
                            "class_name"
                        ].to_numpy()
                        ==
                        class_name
                    )
                )


            records.append(
                record
            )


    output_path = (
        SPLIT_ROOT
        /
        "player_split_assignment.csv"
    )


    pd.DataFrame(
        records
    ).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        f"球员划分表："
        f"{output_path}"
    )


# ============================================================
# 29. 保存本次数据集划分配置
# ============================================================

def save_split_config() -> None:

    config = {
        "random_seed":
            RANDOM_SEED,

        "train_size":
            TRAIN_SIZE,

        "val_size":
            VAL_SIZE,

        "test_size":
            TEST_SIZE,

        "max_split_attempts":
            MAX_SPLIT_ATTEMPTS,

        "minimum_players_per_split":
            MIN_PLAYERS_PER_SPLIT,

        "minimum_samples_per_class":
            MIN_SAMPLES_PER_CLASS,

        "minimum_players_per_class_per_split":
            MIN_PLAYERS_PER_CLASS_PER_SPLIT,

        "class_player_balance_weight":
            CLASS_PLAYER_BALANCE_WEIGHT,

        "player_concentration_weight":
            PLAYER_CONCENTRATION_WEIGHT,

        "class_names":
            CLASS_NAMES,
    }


    output_path = (
        SPLIT_ROOT
        /
        "split_config.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )


    print(
        f"划分配置："
        f"{output_path}"
    )


# ============================================================
# 30. 主程序
# ============================================================

def main() -> None:

    print("=" * 70)

    print(
        "按球员划分篮球投篮数据集"
    )

    print("=" * 70)


    # ========================================================
    # 收集原始视频
    # ========================================================

    dataframe = (
        collect_samples()
    )


    print(
        f"\n原始数据目录："
        f"{RAW_ROOT}"
    )


    print(
        f"总视频数量："
        f"{len(dataframe)}"
    )


    print(
        "不同球员数量："
        f"{dataframe['player_id'].nunique()}"
    )


    # ========================================================
    # 总体类别分布
    # ========================================================

    total_class_counts = (
        dataframe[
            "class_name"
        ]
        .value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0,
        )
    )


    print(
        "\n总体类别视频数量："
    )


    print(
        total_class_counts.to_string()
    )


    # ========================================================
    # 检查重新标注导致的跨类别同名视频
    # ========================================================

    verify_cross_class_filename_conflicts(
        dataframe
    )


    # ========================================================
    # 打印球员类别关系
    # ========================================================

    print_player_class_distribution(
        dataframe
    )


    # ========================================================
    # 检查每个类别球员数量是否足够
    # ========================================================

    verify_class_player_capacity(
        dataframe
    )


    # ========================================================
    # 搜索最佳划分
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "开始搜索满足约束的数据划分"
    )

    print("=" * 70)


    splits = (
        split_by_group(
            dataframe
        )
    )


    # ========================================================
    # 最终完整验证
    # ========================================================

    verify_no_player_leakage(
        splits
    )


    verify_all_classes(
        splits
    )


    verify_minimum_players(
        splits,
        minimum=(
            MIN_PLAYERS_PER_SPLIT
        ),
    )


    verify_minimum_class_samples(
        splits,
        minimum=(
            MIN_SAMPLES_PER_CLASS
        ),
    )


    verify_minimum_players_per_class(
        splits
    )


    verify_unique_target_names(
        splits
    )


    # ========================================================
    # 打印新划分结果
    # ========================================================

    print_statistics(
        splits
    )


    print_split_class_player_distribution(
        splits
    )


    # ========================================================
    # 正式写入磁盘
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "保存新的数据集划分"
    )

    print("=" * 70)


    copy_split_videos(
        splits
    )


    save_split_summary(
        splits
    )


    save_player_split_assignment(
        splits
    )


    save_split_config()


    # ========================================================
    # 完成
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "数据集划分完成"
    )

    print("=" * 70)


    print(
        f"输出目录："
        f"{SPLIT_ROOT}"
    )


    print(
        "\n已生成："
    )


    print(
        "  train/"
    )

    print(
        "  val/"
    )

    print(
        "  test/"
    )

    print(
        "  train.csv"
    )

    print(
        "  val.csv"
    )

    print(
        "  test.csv"
    )

    print(
        "  split_summary.csv"
    )

    print(
        "  player_split_assignment.csv"
    )

    print(
        "  split_config.json"
    )


    print(
        "\n最重要检查："
    )

    print(
        "打开 split_summary.csv，"
        "确认 TEST 中三个类别的 "
        "player_count 都 >= 3。"
    )


    print("=" * 70)


# ============================================================
# 31. 程序入口
# ============================================================

if __name__ == "__main__":

    main()