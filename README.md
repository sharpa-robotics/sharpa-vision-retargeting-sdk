# Sharpa Wave Vision Teleop

Webcam + MediaPipe OR Meta Quest → palm/finger landmark scaling → vector retargeting → Sharpa Wave hand(s).  
Supports **mono** or **bimanual** control.


## Repository layout

```text
vision-teleop/
├── sharpa_wave_webcam.py            # entry point
├── sharpa_wave_quest.py             # entry point
├── requirements.txt
├── README.md
├── Sharpa Wave Vision Based TeleOp Control Documentation.pdf
├── supplements/                       # shared code, YAML, MediaPipe model
│   ├── teleop_hand.py                 # Wave connect, landmark scales, TeleopHand
│   ├── hand_calibration.py
│   ├── hand_visualization.py
│   ├── single_hand_detector.py
│   ├── frame_queue_utils.py
│   ├── sharpa_wave_left.yml
│   ├── sharpa_wave_right.yml
│   └── hand_landmarker.task
├── sharpa-urdf-usd-xml/wave_01/       # URDFs (retarget + wireframe)
│   ├── left_sharpa_wave/
│   └── right_sharpa_wave/
└── calibration/                       # written at runtime (flat-hand JSON)
```

## Pipeline (short)

1. Camera frames → MediaPipe (`single_hand_detector.py`)
2. Flat-hand calibration → establish neutral position (`hand_calibration.py`)
3. Per-finger MCP offset/straightening + apply finger landmark scales → processed hand is sent to IK solver (`teleop_hand.py`)

Per frame: correct → scale landmarks → retarget → command Sharpa Wave
Optional wireframe / overlays (`hand_visualization.py`)


## Where to change paths

Entry script uses `REPO_ROOT` (= this folder) and `SUPPLEMENTS_DIR` (= `./supplements`).  
Modules under `supplements/` use `REPO_ROOT = supplements.parent` for URDF / calibration.

| File | Variable | Default |
|------|----------|---------|
| `sharpa_wave_webcam.py` | `DEFAULT_SHARPA_SDK_PYTHON` | `/opt/sharpa-wave-sdk/python` |
| | `DEFAULT_CONFIG_PATH` | `./supplements/sharpa_wave_left.yml` |
| | `DEFAULT_ROBOT_DIR_PATH` | `./sharpa-urdf-usd-xml/wave_01` |
| | `DEFAULT_CAMERA_PATH` | `/dev/video0` |
| `supplements/teleop_hand.py` | `DEFAULT_URDF_WAVE01` | `./sharpa-urdf-usd-xml/wave_01` |
| | `DEFAULT_CALIBRATION_DIR` | `./calibration` |
| `supplements/hand_calibration.py` | `DEFAULT_CALIBRATION_DIR` | `./calibration` |
| `supplements/single_hand_detector.py` | `DEFAULT_MODEL_PATH` | `./supplements/hand_landmarker.task` |
| `supplements/hand_visualization.py` | `DEFAULT_URDF_WAVE01` | `./sharpa-urdf-usd-xml/wave_01` |

CLI overrides: `--config-path`, `--robot-dir`, `--camera-path`.

**URDF resolution:** `--robot-dir` is usually `wave_01/`. Code resolves to `wave_01/{left,right}_sharpa_wave/`, then:

- Retargeting loads the YAML filename (e.g. `left_sharpa_wave_with_wrist.urdf`) under that side folder via `RetargetingConfig.set_default_urdf_dir`.
If using a different URDF, adjust the file name within the YAML and `hand_visualization.py` 

The default hand control, if a side is not specified, is left  

## Manual setup on a new machine

1. Install Sharpa Wave SDK; set `DEFAULT_SHARPA_SDK_PYTHON` if not `/opt/sharpa-wave-sdk/python`.
2. Clone / copy this `sharpa-vision-retargeting-sdk` tree (include `supplements/`, `sharpa-urdf-usd-xml/`, and `hand_landmarker.task`).
3. Install dependencies:
   ```bash
   cd vision-teleop
   pip install -r requirements.txt
   ```
4. Ensure `pinocchio` (`import pin`) works — required for retargeting and wireframe. If pip cannot install `pin`, use conda-forge.

## Notes about requirements

`pip install -r requirements.txt` installs MediaPipe, OpenCV, torch, matplotlib, and **`dex-retargeting`** from GitHub (`dexsuite/dex-retargeting@v0.5.0`).

**Install separately (not reliably on public PyPI):**

| Package | Notes |
|---------|--------|
| `sharpa` | Proprietary SDK under `DEFAULT_SHARPA_SDK_PYTHON` |
| `pinocchio` (`pin`) | Needed even if dex-retargeting installs; conda-forge if pip fails |
| `nlopt` | Usually pulled with dex-retargeting; required for the vector optimizer |

Also: OpenCV needs a display for mode selection and optional visualization

## Run

From this `vision-teleop/` directory:

```bash
cd /opt/sharpa-wave-sdk/vision-teleop   # or your clone path

# Explicit configs  #run sharpa_wave_quest.py with the same configurations if using Meta  Quest
python3 sharpa_wave_webcam.py \
  --config-path supplements/sharpa_wave_left.yml

python3 sharpa_wave_webcam.py \
  --config-path supplements/sharpa_wave_right.yml --camera-path /dev/video0

# Bimanual
python3 sharpa_wave_webcam.py \
  --config-path supplements/sharpa_wave_left.yml supplements/sharpa_wave_right.yml --robot-dir ./sharpa-urdf-usd-xml/wave_01 --camera-path /dev/video0
```

Modes after calibration: `1` viz+control, `2` viz only, `3` control only (lowest latency).

## Calibration

Flat-hand prompt writes `calibration/hand_calibration_{left,right}.json`. Recalibrate per operation.  
After calibration, terminal prints frozen palm and per-finger scale factors.

## Scaling
If YAML `scaling_factor` is set to `1.0`, it is overridden in teleop_hand.py by the custom landmark scale factors calculated during calibration -- the palm and each finger is scaled independently
If `scaling_factor` is set to any other value, that will be the scaling_factor used -- the custom scale factors will be bypassed

## Parallel retargeting

Dual mode runs left/right `solve_from_joint_pos` on a thread pool, then commands both from the same MediaPipe frame.

## License / third party

Follow Sharpa, MediaPipe, and `dex-retargeting` licensing for SDK, URDFs, and the `.task` model.
