from __future__ import annotations

import filecmp
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(r"F:\BasketballVideoMAE")

# 每次 X-AnyLabeling 独立导出的根目录
EXPORT_ROOT = PROJECT_ROOT / "data" / "xany_exports"

# 汇总后的三分类视频目录
RAW_BY_CLASS_ROOT = PROJECT_ROOT / "data" / "raw_by_class"

# 单独保存每次导出的 metadata
METADATA_ARCHIVE_ROOT = (
    PROJECT_ROOT / "data" / "xany_metadata"
)

# 全项目只保留一份标准 label_map
CANONICAL_LABEL_MAP = (
    PROJECT_ROOT / "data" / "label_map.txt"
)

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


def prepare_directories() -> None:
    RAW_BY_CLASS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_ARCHIVE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for class_name in CLASS_NAMES:
        class_dir = (
            RAW_BY_CLASS_ROOT / class_name
        )
        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # 项目级标签文件固定由 CLASS_NAMES 生成，
    # 不复制任何单次导出的 label_map.txt。
    canonical_content = (
        "\n".join(CLASS_NAMES) + "\n"
    )

    CANONICAL_LABEL_MAP.write_text(
        canonical_content,
        encoding="utf-8",
    )

    print(
        "项目标准标签文件："
        f"{CANONICAL_LABEL_MAP}"
    )


def validate_player_filename(video_path: Path) -> None:
    """
    要求文件名以 P001_、P002_ 等形式开头，
    这样 split_by_player.py 才能正确读取球员编号。
    """
    player_id = video_path.stem.split("_")[0]

    if re.fullmatch(r"P\d+", player_id) is None:
        raise ValueError(
            f"文件名没有以球员编号开头：{video_path.name}\n"
            "请确保原始视频命名类似："
            "P001_session01.mp4"
        )


def read_label_map(path: Path) -> list[str]:
    """读取标签文件，忽略空行和 UTF-8 BOM。"""
    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]


def handle_label_map(export_dir: Path) -> None:
    """
    检查当前导出批次的标签是否合法。

    允许：
    - 当前批次只包含一个类别；
    - 当前批次只包含两个类别；
    - 标签顺序与 CLASS_NAMES 不同。

    不允许：
    - 出现未知标签；
    - 标签拼写错误；
    - label_map.txt 中存在重复标签。
    """
    source_label_map = export_dir / "label_map.txt"

    if not source_label_map.exists():
        print(
            f"警告：缺少 label_map.txt："
            f"{source_label_map}"
        )
        return

    current_labels = read_label_map(source_label_map)

    if not current_labels:
        raise RuntimeError(
            f"label_map.txt 为空：{source_label_map}"
        )

    # 检查是否有重复标签
    if len(current_labels) != len(set(current_labels)):
        raise RuntimeError(
            "label_map.txt 中存在重复标签：\n"
            f"{source_label_map}\n"
            f"内容：{current_labels}"
        )

    allowed_labels = set(CLASS_NAMES)
    unknown_labels = set(current_labels) - allowed_labels

    if unknown_labels:
        raise RuntimeError(
            "发现未知或拼写错误的标签：\n"
            f"文件：{source_label_map}\n"
            f"未知标签：{sorted(unknown_labels)}\n"
            f"允许标签：{CLASS_NAMES}"
        )

    print(
        "标签检查通过："
        f"{current_labels}"
    )


def archive_metadata(export_dir: Path) -> None:
    source_metadata = export_dir / "metadata.json"

    if not source_metadata.exists():
        print(
            f"警告：缺少 metadata.json："
            f"{export_dir}"
        )
        return

    target_metadata = (
        METADATA_ARCHIVE_ROOT
        / f"{export_dir.name}_metadata.json"
    )

    if target_metadata.exists():
        # 内容相同则跳过，内容不同则报错
        if filecmp.cmp(
            source_metadata,
            target_metadata,
            shallow=False,
        ):
            print(
                "元数据已经归档，跳过："
                f"{target_metadata.name}"
            )
            return

        raise RuntimeError(
            "目标元数据已存在，但内容不同：\n"
            f"源文件：{source_metadata}\n"
            f"目标文件：{target_metadata}\n"
            "可能重新导出了同一个视频，"
            "请确认是否应更新归档。"
        )

    shutil.copy2(
        source_metadata,
        target_metadata,
    )

    print(
        "已归档元数据："
        f"{target_metadata.name}"
    )


def collect_videos(export_dir: Path) -> int:
    videos_root = export_dir / "videos"

    if not videos_root.exists():
        print(f"跳过，没有 videos 文件夹：{export_dir}")
        return 0

    copied_count = 0

    for class_name in CLASS_NAMES:
        source_class_dir = videos_root / class_name

        if not source_class_dir.exists():
            print(
                f"提示：{export_dir.name} "
                f"本次没有类别 {class_name}"
            )
            continue

        target_class_dir = (
            RAW_BY_CLASS_ROOT / class_name
        )

        for source_video in sorted(
            source_class_dir.rglob("*")
        ):
            if (
                not source_video.is_file()
                or source_video.suffix.lower()
                not in VIDEO_EXTENSIONS
            ):
                continue

            validate_player_filename(source_video)

            target_video = (
                target_class_dir / source_video.name
            )

            # 把安全重复运行逻辑放在这里
            if target_video.exists():
                if filecmp.cmp(
                    source_video,
                    target_video,
                    shallow=False,
                ):
                    print(
                        "视频已经汇总，跳过："
                        f"{target_video.name}"
                    )
                    continue

                raise RuntimeError(
                    "发现同名但内容不同的视频：\n"
                    f"源文件：{source_video}\n"
                    f"目标文件：{target_video}"
                )

            shutil.copy2(
                source_video,
                target_video,
            )

            copied_count += 1

    return copied_count


def main() -> None:
    if not EXPORT_ROOT.exists():
        raise FileNotFoundError(
            f"导出根目录不存在：{EXPORT_ROOT}"
        )

    prepare_directories()

    export_dirs = sorted(
        path
        for path in EXPORT_ROOT.iterdir()
        if path.is_dir()
    )

    if not export_dirs:
        raise RuntimeError(
            f"没有找到任何导出目录：{EXPORT_ROOT}"
        )

    total_copied = 0

    for index, export_dir in enumerate(
        export_dirs,
        start=1,
    ):
        print("\n" + "=" * 60)
        print(
            f"[{index}/{len(export_dirs)}] "
            f"处理：{export_dir.name}"
        )

        handle_label_map(export_dir)
        archive_metadata(export_dir)

        copied_count = collect_videos(export_dir)
        total_copied += copied_count

        print(f"复制视频数量：{copied_count}")

    print("\n" + "=" * 60)
    print("汇总完成")
    print(f"总复制视频数：{total_copied}")
    print(f"训练原始数据：{RAW_BY_CLASS_ROOT}")
    print(f"元数据归档：{METADATA_ARCHIVE_ROOT}")
    print(f"标准标签文件：{CANONICAL_LABEL_MAP}")


if __name__ == "__main__":
    main()