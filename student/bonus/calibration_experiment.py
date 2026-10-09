"""Measure camera yaw-calibration sensitivity without touching graded artifacts.

Run from the repository root with:
    .venv/bin/python student/bonus/calibration_experiment.py

The detector runs once per frame. Separate trackers receive identical LiDAR
detections and identical seeded camera pixels, with only camera yaw changed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform"))
sys.path.insert(0, str(ROOT / "platform/third_party/waymo_reader"))

from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils

from fusion_lab import tracking_params
from fusion_lab.evaluation import tracking_counts, valid_ground_truth
from fusion_lab.lidar_pcl import pcl_from_range_image
from fusion_lab.scripts.run_lab import _lidar_observations
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace


SEGMENT = ROOT / "data/Waymo/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"
WEIGHTS = ROOT / "data/weights/pretrained_fpn-resnet/fpn_resnet_18_epoch_300.pth"
REFERENCE_LOG = ROOT / "student/artifacts/grade_run_fused.log"
OUTPUT = ROOT / "student/bonus/calibration_results.json"
YAW_DEGREES = (0.0, 1.0, 3.0, 5.0)
FIRST_FRAME, LAST_FRAME, SEED = 0, 198, 0


class RecordingFilter(Filter):
    """Record residuals of camera observations actually accepted by association."""

    def __init__(self, kalman_module):
        super().__init__(kalman_module)
        self.camera_innovation_norms: list[float] = []

    def update(self, track, meas):
        if meas.sensor.name == "camera":
            residual = np.asarray(self._k.innovation(track.x, meas), dtype=float)
            self.camera_innovation_norms.append(float(np.linalg.norm(residual)))
        super().update(track, meas)


def yaw_perturbed_sensor(calibration, camera_module, degrees: float) -> Sensor:
    """Premultiply sensor-to-vehicle extrinsic by a vehicle-frame yaw rotation."""
    sensor = Sensor("camera", calibration, camera_module)
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.eye(4)
    rotation[:2, :2] = ((cosine, -sine), (sine, cosine))
    sensor_to_vehicle = rotation @ np.asarray(sensor.sens_to_veh, dtype=float)
    sensor.sens_to_veh = np.asmatrix(sensor_to_vehicle)
    sensor.veh_to_sens = np.asmatrix(np.linalg.inv(sensor_to_vehicle))
    return sensor


def camera_pixels(frame, rng) -> list[np.ndarray] | None:
    """Use runner's FRONT GT centres and one shared seeded noise draw per label."""
    group = next((g for g in frame.camera_labels
                  if g.name == dataset_pb2.CameraName.FRONT), None)
    if group is None:
        return None
    return [np.array([label.box.center_x, label.box.center_y])
            + rng.normal(0, 0.5, 2)
            for label in group.labels
            if label.type == label_pb2.Label.Type.TYPE_VEHICLE]


def probe_camera_candidates(tracks, measurements, sensor, association, kalman):
    """Inspect the best Mahalanobis candidate for each visible confirmed track.

    These diagnostics precede greedy association. A track is gate-rejected when
    it has observations but no candidate inside the chi-square gate.
    """
    residuals = []
    rejected = 0
    for track in tracks:
        if track.state != "confirmed" or not sensor.in_fov(track.x) or not measurements:
            continue
        candidates = []
        for measurement in measurements:
            distance = association.mahalanobis_distance(track, measurement)
            gamma = np.asarray(kalman.innovation(track.x, measurement), dtype=float)
            candidates.append((distance, float(np.linalg.norm(gamma))))
        best_distance, best_residual = min(candidates, key=lambda item: item[0])
        residuals.append(best_residual)
        rejected += not association.chi2_gate(best_distance, sensor)
    return residuals, rejected


def main() -> None:
    for required in (SEGMENT, WEIGHTS, REFERENCE_LOG):
        if not required.is_file():
            raise FileNotFoundError(required)
    reference = {row["frame"]: row for row in
                 map(json.loads, REFERENCE_LOG.read_text().splitlines())}
    ws = load_workspace()
    bev, det, association = (ws[key] for key in
                             ("bev_mapping", "detection_pipeline", "association"))
    config = det.load_fpn_resnet_config(str(WEIGHTS))
    model = det.create_fpn_model(config, str(WEIGHTS))
    managers = {degrees: TrackManager(ws["track_management"])
                for degrees in YAW_DEGREES}
    filters = {degrees: RecordingFilter(ws["kalman"])
               for degrees in YAW_DEGREES}
    stats = {degrees: {"matches": 0, "sum_sq_err": 0.0, "ghosts": 0,
                       "misses": 0, "camera_updates": 0,
                       "candidate_probes": 0, "gate_rejected_tracks": 0,
                       "best_candidate_innovation_sum_px": 0.0}
             for degrees in YAW_DEGREES}
    lidar_sensor = None
    camera_sensors = None
    rng = np.random.default_rng(SEED)
    frames_processed = 0

    for frame_no, frame in enumerate(WaymoDataFileReader(str(SEGMENT))):
        if frame_no < FIRST_FRAME:
            continue
        if frame_no > LAST_FRAME:
            break
        if lidar_sensor is None:
            lidar_calib = waymo_utils.get(frame.context.laser_calibrations,
                                          dataset_pb2.LaserName.TOP)
            camera_calib = waymo_utils.get(frame.context.camera_calibrations,
                                           dataset_pb2.CameraName.FRONT)
            lidar_sensor = Sensor("lidar", lidar_calib, ws["camera_fusion"])
            camera_sensors = {degrees: yaw_perturbed_sensor(camera_calib,
                              ws["camera_fusion"], degrees)
                              for degrees in YAW_DEGREES}

        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, config)).unsqueeze(0).float()
        detections = det.detect_objects_from_bev(tensor, model, config)
        labels = valid_ground_truth(frame.laser_labels, config,
                                    label_pb2.Label.Type.TYPE_VEHICLE)
        lidar_measurements = _lidar_observations(frame_no, detections,
                                                 lidar_sensor, config)
        pixels = camera_pixels(frame, rng)

        for degrees in YAW_DEGREES:
            manager = managers[degrees]
            filter_obj = filters[degrees]
            camera_sensor = camera_sensors[degrees]
            for track in manager.track_list:
                filter_obj.predict(track)
                track.set_t(frame_no * tracking_params.dt)
            association.associate_and_update(manager, lidar_measurements,
                                             filter_obj, lidar_sensor)
            if pixels is not None:
                camera_measurements = []
                for pixel in pixels:
                    camera_sensor.generate_measurement(frame_no, pixel,
                                                       camera_measurements)
                residuals, rejected = probe_camera_candidates(
                    manager.track_list, camera_measurements, camera_sensor,
                    association, ws["kalman"])
                result = stats[degrees]
                result["candidate_probes"] += len(residuals)
                result["gate_rejected_tracks"] += rejected
                result["best_candidate_innovation_sum_px"] += sum(residuals)
                association.associate_and_update(manager, camera_measurements,
                                                 filter_obj, camera_sensor)

            counts = tracking_counts(manager.track_list, labels)
            result = stats[degrees]
            for source, destination in (("matches", "matches"),
                                        ("sum_sq_err", "sum_sq_err"),
                                        ("ghosts", "ghosts"),
                                        ("misses", "misses")):
                result[destination] += counts[source]
            if degrees == 0.0:
                original = reference[frame_no]
                for key in ("matches", "ghosts", "misses", "confirmed"):
                    if counts[key] != original[key]:
                        raise AssertionError((frame_no, key, counts[key], original[key]))
                if not math.isclose(counts["sum_sq_err"], original["sum_sq_err"],
                                    abs_tol=1e-8):
                    raise AssertionError((frame_no, "sum_sq_err"))
        frames_processed += 1
        if frame_no % 25 == 0:
            print(f"Processed frame {frame_no}/{LAST_FRAME}", flush=True)

    if frames_processed != LAST_FRAME - FIRST_FRAME + 1:
        raise AssertionError(f"Expected 199 frames, got {frames_processed}")

    rows = []
    for degrees in YAW_DEGREES:
        values = stats[degrees]
        probes = values["candidate_probes"]
        matches = values["matches"]
        rows.append({
            "yaw_offset_degrees": degrees,
            "rmse_m": math.sqrt(values["sum_sq_err"] / matches) if matches else None,
            "matches": matches,
            "sum_sq_err_m2": values["sum_sq_err"],
            "ghost_track_frames": values["ghosts"],
            "missed_gt_frames": values["misses"],
            "camera_updates": len(filters[degrees].camera_innovation_norms),
            "mean_accepted_innovation_px": (
                float(np.mean(filters[degrees].camera_innovation_norms))
                if filters[degrees].camera_innovation_norms else None),
            "candidate_probes": probes,
            "mean_best_candidate_innovation_px": (
                values["best_candidate_innovation_sum_px"] / probes if probes else None),
            "gate_rejected_tracks": values["gate_rejected_tracks"],
            "gate_reject_fraction": (
                values["gate_rejected_tracks"] / probes if probes else None),
        })
    report = {
        "segment": SEGMENT.name,
        "frames": [FIRST_FRAME, LAST_FRAME],
        "seed": SEED,
        "perturbation": "T_sensor_to_vehicle = Rz(vehicle yaw) @ T_original",
        "diagnostic_definition": (
            "For each confirmed track visible to camera with >=1 FRONT observation, "
            "select the measurement having minimum squared Mahalanobis distance. "
            "Record its pixel innovation norm and whether every candidate fails "
            "the chi-square gate. Accepted innovation is measured immediately "
            "before each actual camera EKF update."
        ),
        "baseline_matches_graded_log": True,
        "results": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))
    print("Saved", OUTPUT)
    for row in rows:
        print(f"yaw={row['yaw_offset_degrees']:>3g}°  RMSE={row['rmse_m']:.4f} m  "
              f"matches={row['matches']}  innovation={row['mean_best_candidate_innovation_px']:.1f} px  "
              f"gate_reject={row['gate_reject_fraction']:.1%}")


if __name__ == "__main__":
    main()
