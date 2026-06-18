#!/usr/bin/env python3
"""
PROMETHEUS Demo Video Recorder

Records a cinematic demo video of the autonomous surgical suturing procedure.
Output: MP4 file ready for competition submission.

Usage:
  python record_demo.py
  python record_demo.py -o my_demo.mp4 --time 120
"""

import argparse
from prometheus.simulation import PrometheusSimulation


def main():
    parser = argparse.ArgumentParser(
        description="Record PROMETHEUS demo video for FFAI Robothon submission"
    )
    parser.add_argument("-o", "--output", default="prometheus_demo.mp4",
                        help="Output video file (default: prometheus_demo.mp4)")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Maximum recording time in seconds (default: 60)")

    args = parser.parse_args()

    print("=" * 60)
    print("  🎬 PROMETHEUS Demo Video Recorder")
    print("  FFAI Robothon Summer 2026")
    print("=" * 60)

    sim = PrometheusSimulation()
    sim.run_recording(output_path=args.output, max_time=args.time)

    print("\n🏁 Recording complete. Submit this video with your PR!")


if __name__ == "__main__":
    main()
