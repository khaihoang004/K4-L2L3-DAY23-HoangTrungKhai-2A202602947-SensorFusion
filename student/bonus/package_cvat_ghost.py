"""Package a real ghost-track interval as a small CVAT MOT task.

Run: .venv/bin/python student/bonus/package_cvat_ghost.py

Requires prepare_cvat.py to have exported the 100722... segment. Only the
derived BEV images and MOT import ZIP go under ignored data/cvat_bonus/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform"))
sys.path.insert(0, str(ROOT / "platform/third_party/waymo_reader"))

from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2

from fusion_lab.lidar_pcl import pcl_from_range_image
from fusion_lab.workspace_loader import load_workspace


SEGMENT_ID = "10072231702153043603_5725_000_5745_000_with_camera_labels"
SEGMENT = ROOT / f"data/Waymo/training_segment-{SEGMENT_ID}.tfrecord"
TRACKS_JSON = ROOT / f"student/bonus/cvat_tracks_{SEGMENT_ID}.json"
FIRST, LAST = 143, 155
OUTPUT = ROOT / "data/cvat_bonus" / f"training_segment-{SEGMENT_ID}" / "ghost_task"
REVIEW = ROOT / "student/bonus/cvat_ghost_review.json"


def metric_to_pixel(x: float, y: float, config) -> tuple[int, int]:
    column = (y - config.lim_y[0]) / (config.lim_y[1] - config.lim_y[0]) * config.bev_width
    row = (x - config.lim_x[0]) / (config.lim_x[1] - config.lim_x[0]) * config.bev_height
    return int(round(column)), int(round(row))


def main() -> None:
    if not SEGMENT.is_file() or not TRACKS_JSON.is_file():
        raise FileNotFoundError("Run prepare_cvat.py on the 100722... segment first")
    by_frame = {entry["waymo_frame"]: entry
                for entry in json.loads(TRACKS_JSON.read_text())["frames"]}
    chosen = [by_frame[frame_no] for frame_no in range(FIRST, LAST + 1)]
    ghost_frames = [entry["waymo_frame"] for entry in chosen
                    if any(track["ghost"] and track["id"] == 8
                           for track in entry["tracks"])]
    if ghost_frames != list(range(146, 154)):
        raise AssertionError(f"Unexpected ghost frames: {ghost_frames}")

    ws = load_workspace()
    bev = ws["bev_mapping"]
    det = ws["detection_pipeline"]
    config = det.load_fpn_resnet_config()
    image_dir = OUTPUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for frame_no, frame in enumerate(WaymoDataFileReader(str(SEGMENT))):
        if frame_no < FIRST:
            continue
        if frame_no > LAST:
            break
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        bev_map = bev.bev_maps_from_pcl(points, config)
        image = cv2.applyColorMap(
            np.uint8(np.clip(bev_map[2] * 255, 0, 255)), cv2.COLORMAP_BONE
        )
        entry = by_frame[frame_no]
        cv2.putText(image, f"Waymo {frame_no} | valid GT vehicles: {len(entry['gt'])}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        for gt in entry["gt"]:
            u, v = metric_to_pixel(*gt["xy"], config)
            cv2.circle(image, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(image, "GT " + gt["id"][:6], (u + 5, v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        task_frame = frame_no - FIRST + 1
        path = image_dir / f"{task_frame:06d}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 82]):
            raise IOError(f"Could not write {path}")
        written += 1
    if written != LAST - FIRST + 1:
        raise AssertionError(f"Expected 13 BEV images, wrote {written}")

    rows = []
    for entry in chosen:
        task_frame = entry["waymo_frame"] - FIRST + 1
        for track in entry["tracks"]:
            x, y, width, height = track["bbox_xywh"]
            rows.append(f"{task_frame},{track['id'] + 1},{x:.3f},{y:.3f},"
                        f"{width:.3f},{height:.3f},1,1,1.0")
    archive_path = OUTPUT / "annotations_mot.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("gt/labels.txt", "vehicle\n")
        archive.writestr("gt/gt.txt", "\n".join(rows) + "\n")

    review = {
        "segment": SEGMENT.name,
        "waymo_frames": [FIRST, LAST],
        "task_frames_one_based": [1, LAST - FIRST + 1],
        "ghost_waymo_frames": ghost_frames,
        "ghost_task_frames_one_based": [frame_no - FIRST + 1 for frame_no in ghost_frames],
        "platform_track_id": 8,
        "mot_track_id": 9,
        "example": "Waymo frame 146 = task image 000004.jpg; no valid GT vehicle, confirmed track #8 persists",
        "images_directory": str(image_dir.relative_to(ROOT)),
        "annotations_zip": str(archive_path.relative_to(ROOT)),
        "annotation_rows": len(rows),
    }
    REVIEW.write_text(json.dumps(review, indent=2))
    print("Images:", image_dir)
    print("MOT ZIP:", archive_path)
    print("Review:", REVIEW)
    print("Ghost platform track 8 / MOT track 9 at task frames 4–11")


if __name__ == "__main__":
    main()
