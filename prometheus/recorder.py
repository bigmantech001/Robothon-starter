"""
PROMETHEUS Recorder — Video recording with cinematic camera cuts.

Records the simulation to an MP4 file using MuJoCo's offscreen rendering
and imageio for video encoding. Supports multiple camera angles,
HUD overlays, and automatic camera switching.
"""

import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio as iio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

from .config import CameraConfig


class VideoRecorder:
    """
    Records MuJoCo simulation to MP4 video with cinematic camera work.
    """

    def __init__(self, model, data,
                 output_path: str = "prometheus_demo.mp4",
                 config: Optional[CameraConfig] = None):
        """
        Initialize video recorder.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            output_path: Path for the output MP4 file
            config: Camera configuration
        """
        import mujoco

        self.model = model
        self.data = data
        self.config = config or CameraConfig()
        self.output_path = output_path

        # Create offscreen renderer
        self.renderer = mujoco.Renderer(
            model,
            height=self.config.height,
            width=self.config.width,
        )

        # Build lookup of valid camera names in the model
        self._valid_cameras = set()
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                self._valid_cameras.add(name)
        print(f"   Available cameras: {sorted(self._valid_cameras)}")

        # Frame buffer
        self.frames: List[np.ndarray] = []
        self.frame_interval = 1.0 / self.config.fps
        self.last_frame_time = -self.frame_interval  # capture first frame immediately

        # Camera cut tracking
        self.current_camera_idx = 0
        first_cam = self.config.demo_cuts[0][0] if self.config.demo_cuts else "overview"
        self.current_camera = first_cam if first_cam in self._valid_cameras else None

        self.recording = False
        self._render_failures = 0

    def start(self):
        """Start recording."""
        if not HAS_IMAGEIO:
            print("WARNING: imageio not installed. Video recording disabled.")
            print("Install with: pip install imageio[ffmpeg]")
            return

        self.frames = []
        self.last_frame_time = -self.frame_interval
        self.current_camera_idx = 0
        self._render_failures = 0
        self.recording = True
        print(f"🎬 Recording started — {self.config.width}x{self.config.height} @ {self.config.fps}fps")

    def capture_frame(self, sim_time: float, hud_text: str = ""):
        """
        Capture a frame if enough time has passed since the last frame.

        Args:
            sim_time: Current simulation time
            hud_text: Optional HUD text overlay
        """
        if not self.recording or not HAS_IMAGEIO:
            return

        # Check if it's time for a new frame
        if sim_time - self.last_frame_time < self.frame_interval:
            return

        self.last_frame_time = sim_time

        # Update camera based on cut schedule
        self._update_camera(sim_time)

        # Render the frame with robust fallback chain
        frame = self._render_frame()
        if frame is not None:
            self.frames.append(frame)

    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Render a single frame with robust camera fallback.

        Tries: named camera → 'overview' → free camera → skip.
        Returns numpy array or None.
        """
        import mujoco

        # Strategy 1: Use the scheduled named camera
        if self.current_camera and self.current_camera in self._valid_cameras:
            try:
                self.renderer.update_scene(self.data, camera=self.current_camera)
                frame = self.renderer.render()
                if frame is not None and frame.mean() > 5:  # not black
                    return frame.copy()
            except Exception:
                pass

        # Strategy 2: Fall back to 'overview' camera
        if "overview" in self._valid_cameras:
            try:
                self.renderer.update_scene(self.data, camera="overview")
                frame = self.renderer.render()
                if frame is not None and frame.mean() > 5:
                    return frame.copy()
            except Exception:
                pass

        # Strategy 3: Free camera with explicit scene setup
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

        # All strategies failed
        self._render_failures += 1
        if self._render_failures <= 3:
            print(f"⚠️ Frame render failed (attempt {self._render_failures})")
        return None

    def _update_camera(self, sim_time: float):
        """Update the active camera based on the cut schedule."""
        cuts = self.config.demo_cuts
        if not cuts:
            return

        # Find the active camera for the current time
        for i in range(len(cuts) - 1, -1, -1):
            cam_name, cut_time = cuts[i]
            if sim_time >= cut_time:
                if self.current_camera != cam_name:
                    if cam_name in self._valid_cameras:
                        self.current_camera = cam_name
                        self.current_camera_idx = i
                    # If camera not valid, keep previous camera
                break

    def stop_and_save(self) -> Optional[str]:
        """
        Stop recording and save to MP4.

        Returns:
            Path to saved video file, or None if failed.
        """
        if not self.recording or not HAS_IMAGEIO:
            return None

        self.recording = False

        if not self.frames:
            print("⚠️ No frames captured.")
            return None

        print(f"🎞️ Saving {len(self.frames)} frames to {self.output_path}...")
        if self._render_failures > 0:
            print(f"   ({self._render_failures} frames were skipped due to render failures)")

        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Try multiple encoding strategies
        saved = self._try_save_pyav(output)
        if saved:
            return saved

        saved = self._try_save_mimwrite(output)
        if saved:
            return saved

        # Last resort: save frames as PNGs
        return self._save_frames_as_pngs(output)

    def _try_save_pyav(self, output: Path) -> Optional[str]:
        """Try saving with pyav plugin."""
        try:
            writer = iio.imopen(str(output), "w", plugin="pyav")
            writer.write(
                np.stack(self.frames),
                codec="libx264",
                fps=self.config.fps,
            )
            writer.close()
            duration = len(self.frames) / self.config.fps
            print(f"✅ Video saved: {output} ({duration:.1f}s, {len(self.frames)} frames)")
            return str(output)
        except Exception as e:
            print(f"   pyav save failed: {e}")
            return None

    def _try_save_mimwrite(self, output: Path) -> Optional[str]:
        """Try saving with mimwrite fallback."""
        try:
            iio.mimwrite(str(output), self.frames, fps=self.config.fps)
            duration = len(self.frames) / self.config.fps
            print(f"✅ Video saved: {output} ({duration:.1f}s)")
            return str(output)
        except Exception as e:
            print(f"   mimwrite save failed: {e}")
            return None

    def _save_frames_as_pngs(self, output: Path) -> Optional[str]:
        """Last resort: save individual frames."""
        frames_dir = output.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for i, frame in enumerate(self.frames):
            try:
                iio.imwrite(str(frames_dir / f"frame_{i:05d}.png"), frame)
                saved += 1
            except Exception:
                pass
        print(f"📁 Saved {saved}/{len(self.frames)} frames to {frames_dir}/")
        return str(frames_dir)

    def get_stats(self) -> str:
        """Get recording statistics."""
        if not self.frames:
            return "No frames captured"
        duration = len(self.frames) / self.config.fps
        return f"{len(self.frames)} frames, {duration:.1f}s, camera: {self.current_camera}"
