#!/usr/bin/env python3
"""
PROMETHEUS — Run the Autonomous Dual-Arm Dexterous Surgical Suturing Robot

Usage:
  python run.py                    # Interactive viewer with autonomous mode
  python run.py --teleop           # Interactive viewer with teleoperation
  python run.py --record           # Record demo video
  python run.py --headless         # Headless simulation (no GUI)
  python run.py --record -o demo.mp4 --time 90

FFAI Robothon Summer 2026 — Embodied AI Hackathon
"""

from prometheus.simulation import main

if __name__ == "__main__":
    main()
