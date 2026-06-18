# 🏥 PROMETHEUS — Autonomous Dual-Arm Dexterous Surgical Suturing Robot

> **FFAI Robothon Summer 2026** | Built with Antigravity IDE (Gemini)
>
> The first-ever autonomous surgical suturing simulation in MuJoCo — featuring dual 7-DOF robotic arms with 24-DOF anthropomorphic dexterous hands performing bimanual suturing on simulated tissue.

---

## 🚀 Quick Start

```bash
# Install dependencies (one command)
pip install -r requirements.txt

# Run the simulation (interactive viewer + autonomous mode)
python run.py

# Run in teleoperation mode
python run.py --teleop

# Record demo video
python run.py --record -o prometheus_demo.mp4

# Headless simulation (no GUI, for testing)
python run.py --headless
```

**Requirements:** Python 3.9+, MuJoCo 3.1+, NumPy. No GPU needed. Works on all platforms.

---

## 🎯 Project Overview

**PROMETHEUS** is an autonomous surgical suturing robot that performs the complete suture procedure:

1. **Approach** — Right hand reaches for the curved surgical needle on the instrument tray
2. **Grasp** — Fingers close in a precise pinch grip around the needle body
3. **Position** — Left hand moves to retract and stabilize the tissue
4. **Pierce** — Right hand drives the curved needle through the tissue entry point
5. **Drive Through** — Wrist rotation arcs the needle through the tissue
6. **Regrab** — Left hand catches the emerging needle tip with a tripod grip
7. **Pull Thread** — Left hand pulls the needle and attached suture thread through
8. **Tension** — Right hand applies controlled tension to close the wound
9. **Repeat** — Procedure repeats for 4 suture points along the wound

### What Makes This Unique

- **First-ever surgical suturing in MuJoCo** — No one has built this before
- **54 actively controlled DOF** — Dual 7-DOF arms + dual 20-DOF dexterous hands
- **Bimanual coordination** — Left and right hands perform distinct, coordinated roles
- **Complete MJCF model built from scratch** — Not using any pre-existing robot models
- **15-state autonomous task planner** — Full surgical procedure state machine

---

## 🤖 Robot Specifications

| Component | Specification |
|-----------|--------------|
| **Arms** | 2× 7-DOF serial manipulators (shoulder 3-DOF, elbow 1-DOF, wrist 3-DOF) |
| **Hands** | 2× 24-DOF anthropomorphic hands (5 fingers × 4-5 joints each) |
| **Total DOF** | 54 actively controlled joints |
| **Actuators** | 54 position-controlled servos with force limits |
| **Sensors** | 14 joint position, 4 joint velocity, 12 touch, 2 needle tracking |
| **Tendons** | 8 PIP-DIP coupling tendons (anatomically accurate) |
| **Bodies** | 56 rigid bodies across the full assembly |

### Hand Design

Each hand features anatomically-inspired finger kinematics:
- **Thumb** — 4 DOF (CMC flexion/abduction, MCP, IP)
- **Index** — 4 DOF (MCP abduction, MCP flexion, PIP, DIP)
- **Middle** — 4 DOF (same as index)
- **Ring** — 4 DOF (same as index)
- **Little** — 4 DOF (same as index)

PIP-DIP joints are coupled via tendons at a 1:0.67 ratio, mimicking real hand biomechanics.

### Surgical Instruments

- **Curved Needle** — 8-segment arc of capsule geoms with realistic curved geometry
- **Suture Thread** — 6-segment composite rope with natural drape shape
- **Tissue Pads** — Two tissue pads with a wound gap, entry/exit point markers
- **Instrument Tray** — Stainless steel tray for needle storage

---

## 🎮 Controls (Teleoperation Mode)

```
┌─────────────────────────────────────────────────────────┐
│  ARM CONTROL (active arm)                               │
│  W/S  — Move forward/backward     Q/E — Rotate wrist   │
│  A/D  — Move left/right           R/F — Move up/down   │
│  Z/X  — Elbow flex/extend                               │
│                                                         │
│  HAND CONTROL                                           │
│  1 — Open hand        2 — Pinch grip                    │
│  3 — Tripod grip      4 — Power grasp                   │
│  5 — Tissue retraction                                  │
│                                                         │
│  MODE CONTROL                                           │
│  TAB   — Switch active arm (left ↔ right)               │
│  SPACE — Toggle autonomous/teleop mode                  │
│  G     — Start autonomous suturing                      │
│  ESC   — Exit                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
FFAI-Rob/
├── run.py                       # Main entry point
├── record_demo.py               # Demo video recorder
├── requirements.txt             # Dependencies
├── registration.json            # Competition UUID
├── models/
│   └── prometheus.xml           # Complete MJCF model (~800 lines)
└── prometheus/                  # Python package
    ├── __init__.py
    ├── config.py                # Centralized configuration
    ├── simulation.py            # MuJoCo simulation loop
    ├── controller.py            # Task planner + PD control
    ├── teleoperation.py         # Keyboard control handler
    ├── recorder.py              # Video recording utilities
    └── utils.py                 # Math helpers
```

### Control Architecture

```
Task Planner (FSM)           Trajectory Generator        PD Controller
┌──────────────────┐        ┌───────────────────┐       ┌─────────────┐
│ 15 surgical      │ ──→    │ Cubic spline      │ ──→   │ Per-joint   │
│ states with      │        │ interpolation     │       │ position    │
│ sensor-based     │        │ with smooth-step  │       │ control     │
│ transitions      │        │ blending          │       │ + force     │
│                  │        │                   │       │   limits    │
└──────────────────┘        └───────────────────┘       └─────────────┘
```

---

## 🧪 MuJoCo Features Utilized

| Feature | Usage |
|---------|-------|
| **MJCF XML** | Complete custom model, 800+ lines, modular structure |
| **Joints** | 54 revolute/hinge joints with stacked multi-axis shoulders |
| **Actuators** | 54 position servos with force limits |
| **Tendons** | 8 fixed tendons for PIP-DIP finger coupling |
| **Sensors** | Joint position/velocity, touch, frame position/quaternion |
| **Keyframes** | Gravity-balanced initial arm configuration |
| **Collision Filtering** | 32 exclude pairs for self-collision prevention |
| **Solver** | Euler integrator, elliptic friction cone, CG solver |
| **Visual** | Skybox, shadows, materials, reflectance, multiple cameras |
| **Cameras** | 6 named cameras with cinematic recording cuts |

---

## 📊 Task Design

**Goal:** Autonomously close a simulated wound using 4 interrupted sutures.

**Challenge Level:** Extremely high
- Sub-millimeter precision required for needle grasping
- Curved needle driving demands coordinated wrist rotation
- Bimanual coordination between grasping and tissue retraction
- Thread management requires force-controlled pulling
- Sequential multi-stitch planning with state transitions

**Measurability:**
- Stitch count (0-4)
- Needle grasp success rate
- Tissue penetration accuracy
- Thread tension uniformity
- Total procedure time

---

## 🏆 Innovation

This project represents the **first-ever autonomous surgical suturing simulation in MuJoCo**. While dexterous manipulation research has explored cube rotation, object relocation, and tool use, surgical suturing with dexterous hands is an unexplored frontier that combines:

1. **Medical robotics** — A meaningful real-world application
2. **Dexterous manipulation** — Maximum DOF utilization
3. **Bimanual coordination** — Two hands with distinct roles
4. **Sequential task planning** — 15-state procedure
5. **Tool use** — Grasping and manipulating a surgical instrument
6. **Soft interaction** — Careful force control on tissue

---

## 📋 Registration

```json
{
  "uuid": "e4b2f5e1-c86f-48e9-afea-26dd91cac88d",
  "participant_name": "bigmantech",
  "project_name": "PROMETHEUS"
}
```

---

## 📜 License

MIT License — Built for FFAI Robothon Summer 2026.

**Agent Used:** Antigravity IDE (Gemini)
