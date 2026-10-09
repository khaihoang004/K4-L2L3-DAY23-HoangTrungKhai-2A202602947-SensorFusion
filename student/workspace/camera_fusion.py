"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    transform = np.asarray(sensor.veh_to_sens, dtype=float)
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    sensor_position = transform[:3, :3] @ position + transform[:3, 3]
    if not np.isfinite(sensor_position).all():
        return False
    if sensor.name == "camera" and sensor_position[0] <= 1e-6:
        return False
    angle = np.arctan2(sensor_position[1], sensor_position[0])
    return bool(sensor.fov[0] <= angle <= sensor.fov[1])


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    transform = np.asarray(sensor.veh_to_sens, dtype=float)
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    depth, left, up = transform[:3, :3] @ position + transform[:3, 3]
    if not np.isfinite([depth, left, up]).all() or depth <= 1e-6:
        raise ValueError(
            "Camera projection needs finite coordinates and positive depth "
            f"> 1e-6; sensor position={[depth, left, up]}"
        )
    return np.asmatrix(
        [[sensor.c_i - sensor.f_i * left / depth],
         [sensor.c_j - sensor.f_j * up / depth]]
    )


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    return {
        "z": np.asmatrix(np.asarray(z, dtype=float).reshape(2, 1)),
        "R": np.asmatrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2])),
        "sensor": sensor,
    }
