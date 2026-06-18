# 🏥 PROMETHEUS — Autonomous Dual-Arm Dexterous Surgical Suturing Robot

> **FFAI Robothon Summer 2026**

## 🎬 Demo Video

<!-- Demo video will be embedded here or linked below -->
> Video: see PR description for embedded demo

## 📋 Project Overview

PROMETHEUS is an **autonomous surgical suturing simulation** featuring dual 7-DOF robotic arms with 24-DOF anthropomorphic dexterous hands performing bimanual suturing on simulated tissue in MuJoCo.

### What the robot does (autonomously):

1. **Approaches** the curved surgical needle on the instrument tray
2. **Grasps** the needle with a precision pinch grip (right hand)
3. **Positions** the left hand to retract and stabilize tissue
4. **Pierces** tissue — wrist rotation arcs the curved needle through
5. **Regrasps** — left hand catches the emerging needle tip
6. **Pulls** the thread through with bimanual coordination
7. **Tensions** the thread to close the wound
8. **Repeats** for 4 interrupted sutures

## 🚀 Quick Start

```bash
cd submissions/prometheus
pip install -r requirements.txt

# Interactive viewer (autonomous mode)
python run.py

# Headless mode (no GUI, fast)
python run.py --headless --time 60

# Teleoperation mode
python run.py --teleop

# Record demo video
python record_demo.py --time 90 -o demo.mp4
```

**Requirements:** Python 3.9+, MuJoCo 3.1+, NumPy

## 🔧 Technical Specs

| Component | Details |
|-----------|---------|
| **Arms** | 2 × 7-DOF (shoulder yaw/pitch/roll, elbow flex, wrist pronation/flex/deviation) |
| **Hands** | 2 × 20-DOF anthropomorphic (thumb, index, middle, ring, little — 4 joints each) |
| **Total DOF** | 54 actively controlled joints |
| **Controller** | 15-state hierarchical FSM with smooth trajectory interpolation |
| **Hand Poses** | 5 surgical presets: open, pinch, tripod, tissue retraction, power grasp |
| **Physics** | MuJoCo Euler integrator, elliptic cone contacts, CG solver |
| **Sensors** | 12 touch (fingertips + palms), 14 joint position, 4 velocity, needle tracker |
| **Tendons** | 8 PIP-DIP coupling tendons (anatomically accurate 0.67 ratio) |
| **Cameras** | 6 cinematic: overview, top_down, hands_close, right_side, left_side, detail |

## 🏗️ File Structure

```
submissions/prometheus/
├── run.py                    # Main entry point
├── record_demo.py            # Demo video recorder
├── registration.json         # Competition UUID
├── requirements.txt          # Dependencies
├── pyproject.toml             # Project metadata
├── models/
│   └── prometheus.xml        # Full MJCF model (788 lines)
└── prometheus/
    ├── __init__.py
    ├── config.py              # Centralized configuration
    ├── controller.py          # 15-state suturing FSM
    ├── simulation.py          # MuJoCo simulation loop
    ├── recorder.py            # Video recording system
    ├── teleoperation.py       # Keyboard teleoperation
    └── utils.py               # Math utilities
```

## 💡 Innovation

- **First-ever** autonomous surgical suturing in MuJoCo
- **Maximum DOF utilization** — 54 controlled joints with bimanual coordination
- **Medical robotics application** — meaningful real-world impact
- **Complete surgical workflow** — tool grasping → tissue manipulation → wound closure
- **No GPU required** — runs on any platform with Python + MuJoCo

## Registration

```json
{
  "uuid": "e4b2f5e1-c86f-48e9-afea-26dd91cac88d"
}
```
