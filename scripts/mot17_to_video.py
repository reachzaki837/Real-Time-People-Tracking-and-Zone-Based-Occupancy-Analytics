"""Convert an MOT17 image sequence into an MP4 consumable by the notebooks.

Example:
    python scripts/mot17_to_video.py \
        --sequence-dir data/MOT17/train/MOT17-04-FRCNN \
        --output vidp.mp4
"""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="MOT17 sequence directory containing img1/ and seqinfo.ini.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination MP4 path (for example, vidp.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Override the sequence frame rate; defaults to seqinfo.ini.",
    )
    return parser.parse_args()


def read_sequence_fps(sequence_dir: Path, override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--fps must be greater than zero.")
        return override

    config_path = sequence_dir / "seqinfo.ini"
    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f"Missing sequence metadata: {config_path}")

    try:
        fps = config.getfloat("Sequence", "frameRate")
    except (configparser.Error, ValueError) as error:
        raise ValueError(f"Could not read frameRate from {config_path}") from error
    if fps <= 0:
        raise ValueError(f"Invalid frameRate in {config_path}: {fps}")
    return fps


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir.resolve()
    image_dir = sequence_dir / "img1"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG frames found in {image_dir}")

    fps = read_sequence_fps(sequence_dir, args.fps)
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {image_paths[0]}")
    height, width = first_frame.shape[:2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    try:
        for index, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {image_path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame size mismatch at {image_path}: expected {(width, height)}, "
                    f"got {frame.shape[1::-1]}"
                )
            writer.write(frame)
            if index % 100 == 0 or index == len(image_paths):
                print(f"Converted {index}/{len(image_paths)} frames")
    finally:
        writer.release()

    print(f"Wrote {args.output} ({len(image_paths)} frames at {fps:g} FPS)")


if __name__ == "__main__":
    main()
