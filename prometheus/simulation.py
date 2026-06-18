"""
PROMETHEUS Simulation — Core MuJoCo simulation loop with viewer integration.

This is the main simulation engine that:
  1. Loads the MJCF model
  2. Initializes the controller
  3. Runs the simulation loop
  4. Handles visualization (viewer or offscreen recording)
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

# Fix Windows encoding for emoji/unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import mujoco
import mujoco.viewer

from .config import PrometheusConfig, DEFAULT_CONFIG, MODEL_PATH
from .controller import SuturingController, TaskState
from .teleoperation import TeleoperationHandler
from .recorder import VideoRecorder
from .utils import format_time


class PrometheusSimulation:
    """
    Main simulation class for the PROMETHEUS surgical robot.

    Manages the MuJoCo simulation, controller, visualization,
    and optional video recording.
    """

    def __init__(self, config: PrometheusConfig = DEFAULT_CONFIG):
        """
        Initialize the simulation.

        Args:
            config: Master configuration object
        """
        self.config = config

        # Resolve model path
        model_path = Path(config.model_path)
        if not model_path.exists():
            # Try relative to this file
            alt_path = Path(__file__).parent.parent / "models" / "prometheus.xml"
            if alt_path.exists():
                model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"MJCF model not found at {config.model_path} or {alt_path}\n"
                    f"Make sure the models/ directory is in the project root."
                )

        print(f"🔧 Loading MJCF model from {model_path}...")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Load initial keyframe (gravity-balanced arm configuration)
        if self.model.nkey > 0:
            key_id = 0  # "home" keyframe
            self.data.qpos[:] = self.model.key_qpos[key_id]
            if self.model.key_ctrl.shape[0] > 0:
                self.data.ctrl[:] = self.model.key_ctrl[key_id]
            mujoco.mj_forward(self.model, self.data)
            print(f"   Loaded keyframe 'home' — arms in balanced pose")

        # Print model statistics
        print(f"   Bodies: {self.model.nbody}")
        print(f"   Joints: {self.model.njnt} ({self.model.nv} DOF)")
        print(f"   Actuators: {self.model.nu}")
        print(f"   Sensors: {self.model.nsensor}")
        print(f"   Geoms: {self.model.ngeom}")
        print(f"   Contacts max: {self.model.nconmax}")
        print(f"✅ Model loaded successfully!")

        # Initialize controller
        self.controller = SuturingController(
            self.model, self.data,
            controller_config=config.controller,
            suturing_config=config.suturing,
        )

        # Teleoperation handler
        self.teleop = TeleoperationHandler(
            self.controller,
            config=config.teleop,
        )

        # Video recorder (initialized on demand)
        self.recorder = None

        # Simulation state
        self.running = False
        self.step_count = 0
        self.control_step = 0

    def run_viewer(self, auto_start: bool = True):
        """
        Run simulation with MuJoCo interactive viewer.

        Args:
            auto_start: Whether to automatically start the suturing procedure
        """
        print("\n🖥️  Starting PROMETHEUS in viewer mode...")
        print("   Controls:")
        print("   SPACE — Toggle autonomous/teleop mode")
        print("   G     — Start autonomous suturing")
        print("   TAB   — Switch active arm")
        print("   1-5   — Hand pose presets")
        print("   WASD  — Move active arm")
        print("   ESC   — Exit")
        print()

        self.running = True

        if auto_start:
            self.controller.start_autonomous()
            print("🤖 Autonomous suturing started!")

        # Define key callback for teleoperation
        def key_callback(keycode):
            self.teleop.handle_key_press(keycode)

        # Launch the MuJoCo viewer
        with mujoco.viewer.launch_passive(
            self.model, self.data,
            key_callback=key_callback,
        ) as viewer:
            # Set initial camera
            viewer.cam.lookat[:] = [0, -0.2, 1.0]
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135

            while viewer.is_running() and self.running:
                step_start = time.time()

                # Control update (every N sim steps)
                if self.step_count % self.config.controller.control_decimation == 0:
                    self.teleop.update_continuous()
                    self.controller.update()
                    self.control_step += 1

                # Step simulation
                mujoco.mj_step(self.model, self.data)
                self.step_count += 1

                # Sync viewer
                viewer.sync()

                # Status display (every 500 steps)
                if self.step_count % 500 == 0:
                    status = self.controller.get_status()
                    sys.stdout.write(
                        f"\r⏱ {status['time']} | "
                        f"State: {status['state']:20s} | "
                        f"Stitch: {status['stitch']} | "
                        f"Mode: {status['mode']}"
                    )
                    sys.stdout.flush()

                # Check completion
                if self.controller.state == TaskState.COMPLETE:
                    if self.controller.get_state_progress() >= 1.0:
                        print("\n\n✅ Suturing procedure complete!")
                        # Keep viewer open for inspection
                        while viewer.is_running():
                            mujoco.mj_step(self.model, self.data)
                            viewer.sync()
                            time.sleep(0.01)
                        break

                # Real-time sync
                elapsed = time.time() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print("\n👋 Viewer closed.")

    def run_headless(self, max_steps: int = 30000, max_time: float = 60.0):
        """
        Run simulation without visualization (for testing).

        Args:
            max_steps: Maximum simulation steps
            max_time: Maximum simulation time (seconds)
        """
        print(f"\n🔇 Running headless simulation (max {max_steps} steps, {max_time}s)...")

        self.controller.start_autonomous()
        self.running = True
        self.step_count = 0

        start_wall_time = time.time()

        while self.running and self.step_count < max_steps:
            # Control update
            if self.step_count % self.config.controller.control_decimation == 0:
                self.controller.update()

            # Step simulation
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1

            # Check time limit
            if self.data.time >= max_time:
                print(f"\n⏰ Time limit reached ({max_time}s)")
                break

            # Check completion
            if (self.controller.state == TaskState.COMPLETE and
                    self.controller.get_state_progress() >= 1.0):
                print(f"\n✅ Suturing procedure complete at t={self.data.time:.1f}s")
                break

            # Progress report
            if self.step_count % 5000 == 0:
                status = self.controller.get_status()
                wall_elapsed = time.time() - start_wall_time
                print(f"   Step {self.step_count}: t={status['time']}, "
                      f"state={status['state']}, stitch={status['stitch']}, "
                      f"wall={wall_elapsed:.1f}s")

        wall_total = time.time() - start_wall_time
        sim_ratio = self.data.time / wall_total if wall_total > 0 else 0
        print(f"\n📊 Simulation complete:")
        print(f"   Steps: {self.step_count}")
        print(f"   Sim time: {self.data.time:.2f}s")
        print(f"   Wall time: {wall_total:.2f}s")
        print(f"   Speed: {sim_ratio:.1f}x real-time")

    def run_recording(self, output_path: str = "prometheus_demo.mp4",
                      max_time: float = 60.0):
        """
        Run simulation and record to video.

        Args:
            output_path: Path for the output MP4 file
            max_time: Maximum recording time (seconds)
        """
        print(f"\n🎬 Recording demo to {output_path}...")

        # Initialize recorder
        self.recorder = VideoRecorder(
            self.model, self.data,
            output_path=output_path,
            config=self.config.camera,
        )
        self.recorder.start()

        # Start autonomous suturing
        self.controller.start_autonomous()
        self.running = True
        self.step_count = 0

        while self.running:
            # Control update
            if self.step_count % self.config.controller.control_decimation == 0:
                self.controller.update()

            # Step simulation
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1

            # Capture frame
            status = self.controller.get_status()
            hud = f"PROMETHEUS | {status['state']} | Stitch {status['stitch']} | {status['time']}"
            self.recorder.capture_frame(self.data.time, hud)

            # Check time limit
            if self.data.time >= max_time:
                break

            # Check completion (continue a bit after for outro)
            if (self.controller.state == TaskState.COMPLETE and
                    self.controller.get_state_progress() >= 1.0 and
                    self.data.time > self.controller.state_start_time + 3.0):
                break

            # Progress
            if self.step_count % 10000 == 0:
                print(f"   Recording: t={self.data.time:.1f}s, "
                      f"frames={len(self.recorder.frames)}, "
                      f"state={status['state']}")

        # Save video
        saved_path = self.recorder.stop_and_save()
        if saved_path:
            print(f"\n🎥 Demo video ready: {saved_path}")
        return saved_path


def main():
    """Main entry point — parse arguments and run simulation."""
    parser = argparse.ArgumentParser(
        description="PROMETHEUS — Autonomous Dual-Arm Dexterous Surgical Suturing Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Run with interactive viewer
  python run.py --teleop           Start in teleoperation mode
  python run.py --record           Record demo video
  python run.py --headless         Run without visualization
  python run.py --record -o demo.mp4 --time 90
        """,
    )
    parser.add_argument("--teleop", action="store_true",
                        help="Start in teleoperation mode (no auto-start)")
    parser.add_argument("--record", action="store_true",
                        help="Record demo video (no viewer)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("-o", "--output", default="prometheus_demo.mp4",
                        help="Output video file path (default: prometheus_demo.mp4)")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Maximum simulation time in seconds (default: 60)")
    parser.add_argument("--steps", type=int, default=30000,
                        help="Maximum simulation steps for headless mode (default: 30000)")

    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print("  🏥 PROMETHEUS — Surgical Suturing Robot")
    print("  FFAI Robothon Summer 2026")
    print("  Dual 7-DOF Arms + 24-DOF Dexterous Hands")
    print("=" * 60)

    # Create simulation
    sim = PrometheusSimulation()

    # Run in selected mode
    if args.record:
        sim.run_recording(output_path=args.output, max_time=args.time)
    elif args.headless:
        sim.run_headless(max_steps=args.steps, max_time=args.time)
    else:
        sim.run_viewer(auto_start=not args.teleop)


if __name__ == "__main__":
    main()
