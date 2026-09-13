from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path(r"F:\BasketballVideoMAE\data\split")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_ALLOWED_FRAME_DIFFERENCE = 3
FRAME_COUNT_TOLERANCE_RATIO = 0.02


def get_allowed_frame_difference(expected_frame_count: int) -> int:
    if expected_frame_count <= 0:
        return MAX_ALLOWED_FRAME_DIFFERENCE
    ratio_difference = int(np.ceil(expected_frame_count * FRAME_COUNT_TOLERANCE_RATIO))
    return max(MAX_ALLOWED_FRAME_DIFFERENCE, ratio_difference)


def check_video(path: Path) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, "无法打开"

    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    decoded = 0
    while True:
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            break
        decoded += 1
    cap.release()

    if decoded == 0:
        return False, "整个视频没有成功解码任何帧"
    if decoded < 8:
        return False, f"实际只能解码 {decoded} 帧"
    if fps <= 0:
        return False, f"FPS异常：{fps}"

    if expected > 0:
        difference = expected - decoded
        allowed = get_allowed_frame_difference(expected)
        if difference > allowed:
            return (
                False,
                f"视频没有完整解码：声明 {expected} 帧，实际 {decoded} 帧，"
                f"允许差值 {allowed} 帧，解码比例 {decoded / expected:.2%}",
            )

    return True, f"完整解码成功，{decoded}帧，{fps:.2f} FPS，{width}×{height}"


def main() -> None:
    videos = [
        p for p in DATA_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    failed: list[tuple[Path, str]] = []
    for index, video_path in enumerate(videos, start=1):
        ok, info = check_video(video_path)
        print(f"[{index}/{len(videos)}] {video_path.name}: {info}")
        if not ok:
            failed.append((video_path, info))

    print("\n" + "=" * 60)
    print(f"视频总数：{len(videos)}")
    print(f"异常视频：{len(failed)}")
    for path, reason in failed:
        print(f"异常：{path}，原因：{reason}")


if __name__ == "__main__":
    main()
