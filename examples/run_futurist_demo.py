from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing demo dependency. Install with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URDF = ROOT / "assets" / "Futurist" / "futurist.urdf"
DEFAULT_OUTPUT = ROOT / "outputs" / "futurist_demo.mp4"
DEFAULT_TRAJECTORY = ROOT / "outputs" / "futurist_trajectory.json"
SIDE_JOINTS = {
    "left": {
        "hip_roll": "idx01_left_hip_roll",
        "hip_pitch": "idx03_left_hip_pitch",
        "shoulder": "idx13_left_arm_joint1",
        "elbow": "idx16_left_arm_joint4",
    },
    "right": {
        "hip_roll": "idx07_right_hip_roll",
        "hip_pitch": "idx09_right_hip_pitch",
        "shoulder": "idx20_right_arm_joint1",
        "elbow": "idx23_right_arm_joint4",
    },
}


def referenced_meshes(urdf_path: Path) -> list[Path]:
    text = urdf_path.read_text(encoding="utf-8")
    mesh_names = re.findall(r'<mesh\s+filename="([^"]+)"', text)
    return [urdf_path.parent / mesh_name for mesh_name in mesh_names]


def missing_meshes(urdf_path: Path) -> list[str]:
    return [path.name for path in referenced_meshes(urdf_path) if not path.exists()]


def mujoco_ready_urdf(source_urdf: Path, temp_dir: Path) -> Path:
    text = source_urdf.read_text(encoding="utf-8")
    compiler = (
        "  <mujoco>\n"
        f'    <compiler meshdir="{source_urdf.parent}" discardvisual="false"/>\n'
        "  </mujoco>\n"
    )
    if "<mujoco>" not in text:
        text = re.sub(r"(<robot[^>]*>\n)", r"\1" + compiler, text, count=1)

    output_path = temp_dir / source_urdf.name
    output_path.write_text(text, encoding="utf-8")
    return output_path


def build_model(urdf_path: Path) -> mujoco.MjModel:
    with tempfile.TemporaryDirectory() as tmp:
        ready_urdf = mujoco_ready_urdf(urdf_path, Path(tmp))
        spec = mujoco.MjSpec.from_file(str(ready_urdf))
        spec.option.timestep = 0.002
        spec.option.gravity = [0.0, 0.0, -9.81]

        base = spec.body("base_link")
        if base is None:
            raise ValueError("Missing base_link body in Futurist URDF")
        base.add_freejoint(name="floating_base_joint")

        world = spec.worldbody
        world.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            rgba=[0.05, 0.06, 0.08, 1.0],
        )
        world.add_light(pos=[0, -1.2, 2.4], dir=[0, 0.35, -1], diffuse=[1, 1, 1])
        world.add_light(pos=[-1.2, 0.8, 1.6], dir=[0.5, -0.3, -1], diffuse=[0.5, 0.55, 0.7])

        return spec.compile()


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return

    qpos_addr = int(model.jnt_qposadr[joint_id])
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = float(np.clip(value, low, high))
    data.qpos[qpos_addr] = value


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, duration_s: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    progress = min(1.0, max(0.0, time_s / max(duration_s, 0.1)))
    wave = math.sin(2.0 * math.pi * 0.8 * time_s)

    data.qpos[0] = -0.10 + 0.20 * progress
    data.qpos[1] = 0.0
    data.qpos[2] = 0.75 + 0.02 * math.sin(2.0 * math.pi * 0.5 * time_s)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Futurist joint names are exported as idxNN_*; keep this demo independent
    # from exact robot dimensions by using gentle, symmetric showcase motions.
    for side, sign in (("left", 1.0), ("right", -1.0)):
        joints = SIDE_JOINTS[side]
        set_joint(model, data, joints["hip_roll"], 0.04 * sign * wave)
        set_joint(model, data, joints["hip_pitch"], 0.10 * math.sin(2.0 * math.pi * time_s))
        set_joint(model, data, joints["shoulder"], 0.35 * sign * wave)
        set_joint(model, data, joints["elbow"], -0.45 + 0.12 * wave)

    set_joint(model, data, "idx27_head_joint1", 0.18 * math.sin(1.5 * time_s))
    set_joint(model, data, "idx28_head_joint2", 0.08 * math.sin(2.0 * time_s))

    mujoco.mj_forward(model, data)


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body in model: {body_name}")
    return data.xpos[body_id].copy().round(5).tolist()


def run_demo(
    *,
    urdf_path: Path,
    video_path: Path,
    trajectory_path: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
) -> dict:
    missing = missing_meshes(urdf_path)
    if missing:
        raise FileNotFoundError(
            f"Futurist URDF references {len(missing)} missing mesh file(s), "
            f"for example: {', '.join(missing[:8])}"
        )

    model = build_model(urdf_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    trajectory: list[dict] = []
    total_frames = int(duration_s * fps)

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        apply_pose(model, data, time_s, duration_s)

        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = body_position(model, data, "base_link")
        camera.distance = 2.1
        camera.azimuth = 135.0 + 8.0 * math.sin(0.5 * time_s)
        camera.elevation = -14.0

        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

        if frame_idx % max(1, fps // 10) == 0:
            trajectory.append(
                {
                    "time_s": round(time_s, 3),
                    "base_pos": body_position(model, data, "base_link"),
                }
            )

    summary = {
        "project": "FF Futurist MuJoCo Test Demo",
        "task": "The Futurist humanoid URDF loads with its mesh assets and performs a deterministic showcase animation.",
        "model": str(urdf_path),
        "source": str(ROOT / "assets" / "Futurist"),
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "duration_s": duration_s,
        "fps": fps,
        "success": True,
        "final_base_pos": body_position(model, data, "base_link"),
        "trajectory_samples": trajectory,
    }

    try:
        iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
    except Exception as exc:
        fallback = video_path.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=fps)
        summary["video"] = str(fallback)
        summary["video_fallback_reason"] = str(exc)

    trajectory_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a MuJoCo demo video using the packaged FF Futurist humanoid URDF."
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--check-assets", action="store_true", help="Only validate that all URDF mesh files exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = missing_meshes(args.urdf)
    if args.check_assets:
        result = {"urdf": str(args.urdf), "missing_mesh_count": len(missing), "missing_meshes": missing}
        print(json.dumps(result, indent=2))
        return 1 if missing else 0

    summary = run_demo(
        urdf_path=args.urdf,
        video_path=args.output,
        trajectory_path=args.trajectory,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "trajectory_samples"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
