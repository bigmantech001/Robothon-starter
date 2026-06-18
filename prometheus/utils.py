"""
PROMETHEUS Utilities — Math helpers, quaternion operations, and physics tools.
"""

import numpy as np
from typing import Tuple, Optional


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (w, x, y, z format — MuJoCo convention).

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.

    Args:
        q1: Start quaternion
        q2: End quaternion
        t: Interpolation parameter [0, 1]
    """
    dot = np.dot(q1, q2)

    # If quaternions are nearly identical, use linear interpolation
    if abs(dot) > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize_quat(result)

    # Ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    if sin_theta < 1e-10:
        return q1

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return normalize_quat(w1 * q1 + w2 * q2)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two arrays."""
    return a + t * (b - a)


def smooth_step(t: float) -> float:
    """
    Smooth step function for trajectory interpolation.
    Maps [0, 1] -> [0, 1] with zero velocity at endpoints.
    Uses Hermite interpolation (3t² - 2t³).
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoother_step(t: float) -> float:
    """
    Even smoother step function with zero acceleration at endpoints.
    Uses quintic interpolation (6t⁵ - 15t⁴ + 10t³).
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(pos1 - pos2))


def get_body_position(data, model, body_name: str) -> np.ndarray:
    """Get the world position of a named body."""
    body_id = model.body(body_name).id
    return data.xpos[body_id].copy()


def get_body_quaternion(data, model, body_name: str) -> np.ndarray:
    """Get the world quaternion of a named body."""
    body_id = model.body(body_name).id
    return data.xquat[body_id].copy()


def get_site_position(data, model, site_name: str) -> np.ndarray:
    """Get the world position of a named site."""
    site_id = model.site(site_name).id
    return data.site_xpos[site_id].copy()


def get_sensor_value(data, model, sensor_name: str) -> np.ndarray:
    """Get the value of a named sensor."""
    sensor_id = model.sensor(sensor_name).id
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr:adr + dim].copy()


def get_contact_forces(data, model, body_name: str) -> Tuple[float, int]:
    """
    Get total contact force magnitude on a body and number of contacts.

    Returns:
        Tuple of (total_force_magnitude, contact_count)
    """
    body_id = model.body(body_name).id
    total_force = 0.0
    contact_count = 0

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_body = model.geom_bodyid[contact.geom1]
        geom2_body = model.geom_bodyid[contact.geom2]

        if geom1_body == body_id or geom2_body == body_id:
            # Get contact force
            force = np.zeros(6)
            import mujoco
            mujoco.mj_contactForce(model, data, i, force)
            total_force += np.linalg.norm(force[:3])
            contact_count += 1

    return total_force, contact_count


def format_time(seconds: float) -> str:
    """Format time as MM:SS.ms."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"
