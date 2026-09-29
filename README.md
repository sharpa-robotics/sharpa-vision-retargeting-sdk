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
├── Sharpa Wave Vision TeleOp Control Documentation.pdf
├── supplements/                       # shared code and YAML
│   ├── teleop_hand.py                 # Wave connect, landmark scales, TeleopHand
│   ├── hand_calibration.py
│   ├── hand_visualization.py
│   ├── single_hand_detector.py
│   ├── frame_queue_utils.py
│   ├── sharpa_wave_left.yml
│   └── sharpa_wave_right.yml
├── scripts/
│   └── download_hand_landmarker.py    # downloads and verifies the MediaPipe model
├── sharpa-urdf-usd-xml/wave_01/       # URDFs (retarget + wireframe)
│   ├── left_sharpa_wave/
│   └── right_sharpa_wave/
└── calibration/                       # written at runtime (flat-hand JSON)
```

## Pipeline (short)

1. Camera frames → MediaPipe (`single_hand_detector.py`)
2. Flat-hand calibration → correction offsets (`hand_calibration.py`)
3. Frozen palm-span + per-finger MCP→tip scales from neutral vs URDF (`teleop_hand.py`)
4. Per frame: correct → scale landmarks → `dex-retargeting` → Sharpa `set_joint_position`
5. Optional wireframe / overlays (`hand_visualization.py`)

YAML `scaling_factor` is overridden to `1.0` at build time so landmark scales alone control size.

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
- Wireframe uses `{side}_sharpa_wave.urdf` in the same folder.

Both files must exist for viz+control modes.

## Manual setup on a new machine

1. Install Sharpa Wave SDK; set `DEFAULT_SHARPA_SDK_PYTHON` if not `/opt/sharpa-wave-sdk/python`.
2. Clone / copy this `vision-teleop/` tree (include `supplements/` and `sharpa-urdf-usd-xml/`). The MediaPipe model is not included in the repository.
3. Install dependencies:
   ```bash
   cd vision-teleop
   pip install -r requirements.txt
   ```
4. Ensure `pinocchio` (`import pin`) works — required for retargeting and wireframe. If pip cannot install `pin`, use conda-forge.
5. Download the official MediaPipe Hand Landmarker model and verify its SHA-256 digest:
   ```bash
   python3 scripts/download_hand_landmarker.py
   ```

The installer downloads `hand_landmarker/hand_landmarker/float16/1` from Google directly to `supplements/hand_landmarker.task`. To verify an existing download without accessing the network, run:

```bash
python3 scripts/download_hand_landmarker.py --check
```

## Notes about requirements

`pip install -r requirements.txt` installs MediaPipe, OpenCV, torch, matplotlib, and **`dex-retargeting`** from GitHub (`dexsuite/dex-retargeting@v0.5.0`).

**Install separately (not reliably on public PyPI):**

| Package | Notes |
|---------|--------|
| `sharpa` | Proprietary SDK under `DEFAULT_SHARPA_SDK_PYTHON` |
| `pinocchio` (`pin`) | Needed even if dex-retargeting installs; conda-forge if pip fails |
| `nlopt` | Usually pulled with dex-retargeting; required for the vector optimizer |

Also: OpenCV needs a display for `imshow` / mode selection; MediaPipe may use GPU.

## Run

From this `vision-teleop/` directory:

```bash
cd /opt/sharpa-wave-sdk/vision-teleop   # or your clone path

# Left only (default)
python3 sharpa_wave_webcam.py

# Explicit configs
python3 sharpa_wave_webcam.py \
  --config-path supplements/sharpa_wave_left.yml

python3 sharpa_wave_webcam.py \
  --config-path supplements/sharpa_wave_right.yml

# Bimanual
python3 sharpa_wave_webcam.py \
  --config-path supplements/sharpa_wave_left.yml supplements/sharpa_wave_right.yml
```

Modes after calibration: `1` viz+control, `2` viz only, `3` control only (lowest latency).

## Calibration

Flat-hand prompt writes `calibration/hand_calibration_{left,right}.json`. Recalibrate per operator / camera.
After calibration, terminal prints frozen palm and per-finger scale factors.

## Parallel retargeting

Dual mode runs left/right `solve_from_joint_pos` on a thread pool, then commands both from the same MediaPipe frame.

## License / third party

Follow Sharpa, MediaPipe, and `dex-retargeting` licensing for SDK, URDFs, and runtime dependencies. The MediaPipe model is not distributed with this repository; the installation script obtains the pinned model directly from its official Google URL and verifies SHA-256 before use.
