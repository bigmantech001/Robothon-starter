"""
PROMETHEUS Recorder - Video recording with cinematic camera cuts.
"""

import numpy as np
from typing import Optional, List
from pathlib import Path

try:
    import imageio.v2 as iio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio as iio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

from .config import CameraConfig


class VideoRecorder:
    def __init__(self, model, data, output_path="prometheus_demo.mp4", config=None):
        import mujoco
        self.model = model
        self.data = data
        self.config = config or CameraConfig()
        self.output_path = output_path
        self.renderer = mujoco.Renderer(model, height=self.config.height, width=self.config.width)
        self._valid_cameras = set()
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                self._valid_cameras.add(name)
        self.frames: List[np.ndarray] = []
        self.frame_interval = 1.0 / self.config.fps
        self.last_frame_time = -self.frame_interval
        self.current_camera_idx = 0
        first_cam = self.config.demo_cuts[0][0] if self.config.demo_cuts else "overview"
        self.current_camera = first_cam if first_cam in self._valid_cameras else None
        self.recording = False
        self._render_failures = 0

    def start(self):
        if not HAS_IMAGEIO:
            print("WARNING: imageio not installed.")
            return
        self.frames = []
        self.last_frame_time = -self.frame_interval
        self._render_failures = 0
        self.recording = True
        print(f"Recording started - {self.config.width}x{self.config.height} @ {self.config.fps}fps")

    def capture_frame(self, sim_time, hud_text=""):
        if not self.recording or not HAS_IMAGEIO:
            return
        if sim_time - self.last_frame_time < self.frame_interval:
            return
        self.last_frame_time = sim_time
        self._update_camera(sim_time)
        frame = self._render_frame()
        if frame is not None:
            self.frames.append(frame)

    def _render_frame(self):
        import mujoco
        if self.current_camera and self.current_camera in self._valid_cameras:
            try:
                self.renderer.update_scene(self.data, camera=self.current_camera)
                frame = self.renderer.render()
                if frame is not None and frame.mean() > 5:
                    return frame.copy()
            except Exception:
                pass
        if "overview" in self._valid_cameras:
            try:
                self.renderer.update_scene(self.data, camera="overview")
                frame = self.renderer.render()
                if frame is not None and frame.mean() > 5:
                    return frame.copy()
            except Exception:
                pass
        try:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0, -0.2, 0.95]
            cam.distance = 1.3
            cam.elevation = -25
            cam.azimuth = 135
            self.renderer.update_scene(self.data, camera=cam)
            frame = self.renderer.render()
            if frame is not None:
                return frame.copy()
        except Exception:
            pass
        self._render_failures += 1
        return None

    def _update_camera(self, sim_time):
        cuts = self.config.demo_cuts
        if not cuts:
            return
        for i in range(len(cuts) - 1, -1, -1):
            cam_name, cut_time = cuts[i]
            if sim_time >= cut_time:
                if self.current_camera != cam_name and cam_name in self._valid_cameras:
                    self.current_camera = cam_name
                    self.current_camera_idx = i
                break

    def stop_and_save(self):
        if not self.recording or not HAS_IMAGEIO:
            return None
        self.recording = False
        if not self.frames:
            print("No frames captured.")
            return None
        print(f"Saving {len(self.frames)} frames to {self.output_path}...")
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = iio.get_writer(str(output), fps=self.config.fps)
            for frame in self.frames:
                writer.append_data(frame)
            writer.close()
            duration = len(self.frames) / self.config.fps
            print(f"Video saved: {output} ({duration:.1f}s, {len(self.frames)} frames)")
            return str(output)
        except Exception as e:
            print(f"Save failed: {e}")
            frames_dir = output.parent / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(self.frames):
                try:
                    iio.imwrite(str(frames_dir / f"frame_{i:05d}.png"), frame)
                except Exception:
                    pass
            return str(frames_dir)

    def get_stats(self):
        if not self.frames:
            return "No frames captured"
        duration = len(self.frames) / self.config.fps
        return f"{len(self.frames)} frames, {duration:.1f}s, camera: {self.current_camera}"
