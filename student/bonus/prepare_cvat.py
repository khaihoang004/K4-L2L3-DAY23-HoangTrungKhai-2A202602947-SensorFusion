"""Prepare a small BEV tracking task for manual CVAT review.

Run: .venv/bin/python student/bonus/prepare_cvat.py

This writes derived BEV images and a MOT 1.1 import ZIP under ignored data/,
plus a small track JSON under student/bonus/ for the rubric. It does not upload
anything to CVAT or change graded artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform"))
sys.path.insert(0, str(ROOT / "platform/third_party/waymo_reader"))

from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils

from fusion_lab import tracking_params
from fusion_lab.evaluation import _partial_assignment, tracking_counts, valid_ground_truth
from fusion_lab.export_cvat import export_tracks_json
from fusion_lab.lidar_pcl import pcl_from_range_image
from fusion_lab.scripts.run_lab import _front_observations, _lidar_observations
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace


DEFAULT_SEGMENT = ROOT / "data/Waymo/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"
SEGMENT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SEGMENT
WEIGHTS = ROOT / "data/weights/pretrained_fpn-resnet/fpn_resnet_18_epoch_300.pth"
REFERENCE_LOG = ROOT / "student/artifacts/grade_run_fused.log"
BONUS = ROOT / "student/bonus"
CVAT_DATA = ROOT / "data/cvat_bonus" / SEGMENT.stem
SEED, LAST_FRAME = 0, 198


def metric_to_pixel(x: float, y: float, config) -> tuple[float, float]:
    """Project vehicle x/y to BEV image column/row."""
    column = (y - config.lim_y[0]) / (config.lim_y[1] - config.lim_y[0]) * config.bev_width
    row = (x - config.lim_x[0]) / (config.lim_x[1] - config.lim_x[0]) * config.bev_height
    return column, row


def track_bev_box(track, config) -> list[float] | None:
    """Axis-aligned 2D envelope of the track's rotated vehicle footprint."""
    x, y = np.asarray(track.x, dtype=float).reshape(-1)[:2]
    heading = np.array([math.cos(track.yaw), math.sin(track.yaw)]) * track.length / 2
    lateral = np.array([-math.sin(track.yaw), math.cos(track.yaw)]) * track.width / 2
    corners = [np.array([x, y]) + a * heading + b * lateral
               for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    pixel_corners = np.array([metric_to_pixel(c[0], c[1], config) for c in corners])
    u1, v1 = np.maximum(np.min(pixel_corners, axis=0), [0, 0])
    u2, v2 = np.minimum(np.max(pixel_corners, axis=0),
                        [config.bev_width - 1, config.bev_height - 1])
    if u2 - u1 < 1 or v2 - v1 < 1:
        return None
    return [float(u1), float(v1), float(u2 - u1), float(v2 - v1)]


def track_gt_matches(tracks, labels) -> tuple[dict[str, int], set[int]]:
    """Use the same 2 m XY gate and cardinality rule as the grading metrics."""
    confirmed = [track for track in tracks if track.state == "confirmed"]
    if not confirmed or not labels:
        return {}, set()
    positions = np.array([np.asarray(track.x).reshape(-1)[:2]
                          for track in confirmed])
    centres = np.array([[label.box.center_x, label.box.center_y]
                        for label in labels])
    distances = np.linalg.norm(positions[:, None] - centres[None, :], axis=2)
    pairs = _partial_assignment(distances, np.isfinite(distances) & (distances <= 2.0))
    return {labels[j].id: int(confirmed[i].id) for i, j in pairs}, {
        int(confirmed[i].id) for i, _ in pairs
    }


def choose_window(events: list[dict], total_frames: int) -> tuple[int, int, dict | None]:
    """Prefer an ID change where both sightings fit in a short CVAT task."""
    if events:
        event = min(events, key=lambda item: (item["gap_frames"], item["frame"]))
        first = max(0, event["previous_frame"] - 3)
        last = min(total_frames - 1, max(event["frame"] + 3, first + 7))
        return first, last, event
    # No claim of a switch: these frames only demonstrate the import workflow.
    return 76, 87, None


def main() -> None:
    graded_segment = SEGMENT == DEFAULT_SEGMENT
    for path in (SEGMENT, WEIGHTS, *( (REFERENCE_LOG,) if graded_segment else () )):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = ({row["frame"]: row for row in
                  map(json.loads, REFERENCE_LOG.read_text().splitlines())}
                 if graded_segment else {})
    ws = load_workspace()
    bev, det, association = (ws[key] for key in
                             ("bev_mapping", "detection_pipeline", "association"))
    config = det.load_fpn_resnet_config(str(WEIGHTS))
    model = det.create_fpn_model(config, str(WEIGHTS))
    filter_obj = Filter(ws["kalman"])
    manager = TrackManager(ws["track_management"])
    rng = np.random.default_rng(SEED)
    lidar_sensor = camera_sensor = None
    records = []
    images = []
    events = []
    previous_by_gt = {}

    for frame_no, frame in enumerate(WaymoDataFileReader(str(SEGMENT))):
        if frame_no > LAST_FRAME:
            break
        if lidar_sensor is None:
            lidar_calib = waymo_utils.get(frame.context.laser_calibrations,
                                          dataset_pb2.LaserName.TOP)
            camera_calib = waymo_utils.get(frame.context.camera_calibrations,
                                           dataset_pb2.CameraName.FRONT)
            lidar_sensor = Sensor("lidar", lidar_calib, ws["camera_fusion"])
            camera_sensor = Sensor("camera", camera_calib, ws["camera_fusion"])

        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        bev_map = bev.bev_maps_from_pcl(points, config)
        tensor = torch.from_numpy(bev_map).unsqueeze(0).float()
        detections = det.detect_objects_from_bev(tensor, model, config)
        labels = valid_ground_truth(frame.laser_labels, config,
                                    label_pb2.Label.Type.TYPE_VEHICLE)
        observations = _lidar_observations(frame_no, detections, lidar_sensor, config)
        for track in manager.track_list:
            filter_obj.predict(track)
            track.set_t(frame_no * tracking_params.dt)
        association.associate_and_update(manager, observations, filter_obj,
                                         lidar_sensor)
        camera_observations = _front_observations(
            frame, frame_no, camera_sensor, rng,
            dataset_pb2.CameraName.FRONT, label_pb2.Label.Type.TYPE_VEHICLE,
        )
        if camera_observations is not None:
            association.associate_and_update(manager, camera_observations,
                                             filter_obj, camera_sensor)

        counts = tracking_counts(manager.track_list, labels)
        if graded_segment:
            original = reference[frame_no]
            for key in ("matches", "ghosts", "misses", "confirmed"):
                if counts[key] != original[key]:
                    raise AssertionError((frame_no, key, counts[key], original[key]))
            if not math.isclose(counts["sum_sq_err"], original["sum_sq_err"], abs_tol=1e-8):
                raise AssertionError((frame_no, "sum_sq_err"))

        gt_to_track, matched_ids = track_gt_matches(manager.track_list, labels)
        for gt_id, track_id in gt_to_track.items():
            previous = previous_by_gt.get(gt_id)
            if previous and previous[0] != track_id:
                events.append({"gt_id": gt_id,
                               "previous_frame": previous[1], "frame": frame_no,
                               "old_track_id": previous[0], "new_track_id": track_id,
                               "gap_frames": frame_no - previous[1]})
            previous_by_gt[gt_id] = (track_id, frame_no)

        tracks = []
        for track in manager.track_list:
            if track.state != "confirmed":
                continue
            box = track_bev_box(track, config)
            if box is None:
                continue
            tracks.append({"id": int(track.id),
                           "matched_gt_id": next((gt for gt, tid in gt_to_track.items()
                                                  if tid == track.id), None),
                           "ghost": track.id not in matched_ids,
                           "bbox_xywh": box,
                           "xyz": np.asarray(track.x).reshape(-1)[:3].astype(float).tolist()})
        records.append({"waymo_frame": frame_no, "tracks": tracks,
                        "gt": [{"id": label.id,
                                "xy": [float(label.box.center_x),
                                       float(label.box.center_y)]} for label in labels]})

        grey = np.uint8(np.clip(bev_map[2] * 255, 0, 255))
        image = cv2.applyColorMap(grey, cv2.COLORMAP_BONE)
        for label in labels:
            u, v = metric_to_pixel(label.box.center_x, label.box.center_y, config)
            centre = (int(round(u)), int(round(v)))
            cv2.circle(image, centre, 4, (0, 255, 255), -1)
            cv2.putText(image, "GT " + label.id[:6],
                        (centre[0] + 5, centre[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError(f"Could not encode frame {frame_no}")
        images.append(encoded.tobytes())
        if frame_no % 25 == 0:
            print(f"Processed frame {frame_no}/{LAST_FRAME}", flush=True)

    if not records or (graded_segment and len(records) != LAST_FRAME + 1):
        raise AssertionError(f"Unexpected frame count: {len(records)}")
    BONUS.mkdir(parents=True, exist_ok=True)
    CVAT_DATA.mkdir(parents=True, exist_ok=True)
    track_export = BONUS / f"cvat_tracks_{SEGMENT.stem.split('-')[1]}.json"
    export_tracks_json(records, track_export)
    first, last, chosen_event = choose_window(events, len(records))
    image_dir = CVAT_DATA / "images"
    image_dir.mkdir(exist_ok=True)
    mot_rows = []
    for index, frame_no in enumerate(range(first, last + 1), start=1):
        (image_dir / f"{index:06d}.jpg").write_bytes(images[frame_no])
        for track in records[frame_no]["tracks"]:
            x, y, w, h = track["bbox_xywh"]
            mot_rows.append(f"{index},{track['id'] + 1},{x:.3f},{y:.3f},"
                            f"{w:.3f},{h:.3f},1,1,1.0")
    annotation_zip = CVAT_DATA / "annotations_mot.zip"
    with ZipFile(annotation_zip, "w", ZIP_DEFLATED) as archive:
        archive.writestr("gt/labels.txt", "vehicle\n")
        archive.writestr("gt/gt.txt", "\n".join(mot_rows) + "\n")
    summary = {"task_waymo_frames": [first, last],
               "task_image_count": last - first + 1,
               "mot_annotation_count": len(mot_rows),
               "id_change_events": events,
               "suggested_event": chosen_event,
               "image_note": "BEV images with GT centres, not raw camera images",
               "baseline_matches_graded_log": graded_segment}
    review_path = BONUS / f"cvat_review_{SEGMENT.stem.split('-')[1]}.json"
    review_path.write_text(json.dumps(summary, indent=2))
    print("Track export:", track_export)
    print("Review summary:", review_path)
    print("Task images:", image_dir)
    print("MOT annotation ZIP:", annotation_zip)
    print("ID change events:", len(events))
    print("Suggested event:", chosen_event)


if __name__ == "__main__":
    main()
