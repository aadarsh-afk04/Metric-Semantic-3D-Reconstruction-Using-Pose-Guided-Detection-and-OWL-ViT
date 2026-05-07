# CP260-2026 · Metric-Semantic 3D Reconstruction of a Desktop PC Scene

> **Course:** CP260-2026 &nbsp;|&nbsp; **Authors:** K. Aadarsh, P. Himaja  
> **Task:** Oriented Bounding Box (OBB) estimation of PC IO-panel ports + Novel View Synthesis  
> **Evaluation:** May 4 2026, 10:00 AM

---

## What This Project Does

Given **16 posed RGB images** of a PC tower (no depth sensor, no point cloud, no CAD model), this pipeline:

1. **Finds exactly where the IO panel is in every frame** — using camera pose geometry instead of a general object detector (YOLO failed in our scene, detecting a whiteboard instead of the PC).
2. **Detects all port types simultaneously** — Ethernet, Power, USB, VGA, HDMI, Audio, DVI, Serial, PS/2 — using OWL-ViT zero-shot detection at an ultra-low threshold. We detect everything because bonus entities are announced only on exam day and re-running inference is not feasible.
3. **Back-projects detections to 3D world coordinates** — using Depth Anything V2 for metric depth, anchored to the known camera height.
4. **Fits Oriented Bounding Boxes** — Open3D statistical outlier removal cleans up false positives, then OBBs are fitted per entity.
5. **Synthesises novel views** — 3D Gaussian Splatting trained on all 16 frames.

### Why the threshold is so low (`OWL_THRESHOLD = 0.02`)

PC IO ports are **15–50 px** in 2560×1440 images. OWL-ViT confidence scores for objects this small are inherently noisy — the same physical port gets wildly different scores across frames depending on angle and lighting. Raising the threshold causes **missed ports** (false negatives), which are unrecoverable — no 3D points, no OBB. False positives are harmless: they back-project to geometrically scattered locations and are eliminated by Open3D's statistical outlier filter before OBB fitting. **False negatives are fatal. False positives are cheap.**

---

## Directory Structure

```
cp260/
├── Depth-Anything-V2/          ← cloned from GitHub (Step 3)
├── data/
│   ├── frames/
│   │   ├── frame_000319.png
│   │   ├── frame_000333.png
│   │   └── ... (16 frames total, up to frame_000531.png)
│   ├── poses.json              ← 4×4 camera-to-world matrices
│   └── intrinsic.json          ← pinhole camera intrinsics
├── results/                    ← created automatically on first run
│   ├── detections.json         ← raw per-frame detections
│   ├── 3d_detections.png       ← 3D scatter plot
│   ├── preview_<fid>.jpg       ← annotated frame (cyan=crop, green=ports)
│   └── io_zoom_<fid>.jpg       ← zoomed IO panel crop (what OWL-ViT sees)
├── src/
│   └── pipeline.py             ← main pipeline (this repo)
├── answers.json                ← final OBB submission output
└── README.md
```

---

## Camera & Dataset Specs

| Property | Value |
|----------|-------|
| Resolution | 2560 × 1440 px |
| Frames | 16 (frame_000319 → frame_000531, non-consecutive) |
| Distortion | Zero (pre-undistorted pinhole) |
| fx / fy | 1477.010 / 1480.442 px |
| cx / cy | 1298.250 / 686.820 px |
| Camera height above desk | ~0.83 m (used as depth scale anchor) |
| Poses | 4×4 float64 camera-to-world matrices |
| Frames skipped | 400, 531 (IO panel projects below image boundary) |

**Ground-truth VGA socket** (provided for validation only):
```json
{
  "entity": "vga_socket",
  "obb": {
    "center":   [0.2705, 0.2261, 0.8349],
    "extent":   [0.03538, 0.01182, 0.00613],
    "rotation": [[-0.004, 0.9673, -0.2538],
                 [ 0.016, 0.2538,  0.9671],
                 [ 1.000, -0.0001, -0.0163]]
  }
}
```

---

## Setup — Step by Step

### Step 1 — Clone this repository

```bash
git clone https://github.com/<your-username>/cp260-2026.git
cd cp260-2026
mkdir -p data/frames results
```

### Step 2 — Create the conda environment

```bash
conda create -n cp260 python=3.11 -y
conda activate cp260
```

### Step 3 — Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# (use cu121 if you have CUDA 12.1, or remove --index-url for CPU-only)

pip install \
  opencv-python \
  huggingface_hub \
  timm \
  open3d==0.19.0 \
  matplotlib \
  transformers \
  Pillow \
  numpy
```

### Step 4 — Clone Depth Anything V2

```bash
# Run this from inside the cp260/ project root
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

> The model weights (~100 MB) are downloaded automatically from HuggingFace
> on first run. No manual download needed.

### Step 5 — Add your data

Place the 16 frame images into `data/frames/`:

```
data/frames/frame_000319.png
data/frames/frame_000333.png
...
data/frames/frame_000531.png
```

Place `poses.json` and `intrinsic.json` into `data/`.

### Step 6 — Run the pipeline

```bash
conda activate cp260
cd ~/cp260          # or wherever you cloned the repo
python src/pipeline.py
```

Expected runtime: **~3–8 minutes** on GPU, ~20–40 minutes on CPU.

---

## What Happens When You Run

The pipeline processes frames in order of quality (closest/most-centred first):

```
Frame 390 → port projected to (1288, 719) depth=0.24m  ← best frame
Frame 515 → port projected to (960, 548)  depth=0.20m  ← best frame
Frame 371 → port projected to (1594, 274) depth=0.32m
...
Frame 400 → [SKIPPED — port below image boundary]
Frame 531 → [SKIPPED — port below image boundary]
```

For each usable frame:
1. Projects the known VGA world anchor into pixel coordinates (no detector needed)
2. Crops a 640×640 window around the projected location
3. Upscales crop at 8× and 12× and runs OWL-ViT for all port labels
4. Merges detections with NMS (IoU > 0.4)
5. Back-projects each detection to 3D using metric depth

After all frames:
- Aggregates 3D points per entity label
- Removes statistical outliers (k=30, σ=1.5) — this eliminates false positives
- Fits OBB to each inlier cluster with ≥10 points
- Writes `answers.json` and `results/3d_detections.png`

---

## Output Files

### `answers.json`

```json
[
  {
    "entity": "ethernet_socket",
    "obb": {
      "center":   [x, y, z],
      "extent":   [dx, dy, dz],
      "rotation": [[r00, r01, r02],
                   [r10, r11, r12],
                   [r20, r21, r22]]
    }
  },
  { "entity": "power_socket",  "obb": { "..." } },
  { "entity": "usb_port",      "obb": { "..." } },
  { "entity": "vga_socket",    "obb": { "..." } },
  { "entity": "hdmi_port",     "obb": { "..." } },
  { "entity": "audio_jack",    "obb": { "..." } }
]
```

All coordinates are in **metres, world frame** (same coordinate system as `poses.json`).

### `results/preview_<fid>.jpg`

Annotated full-resolution frame:
- **Cyan crosshair** — projected port location (from pose geometry)
- **Cyan rectangle** — crop region sent to OWL-ViT
- **Green rectangles** — detected ports with label and confidence

### `results/io_zoom_<fid>.jpg`

3× upscaled view of the crop that OWL-ViT processed. Use this to visually verify detections are landing on real ports.

---

## Detected Port Types

| Entity key | Description | Typical size |
|------------|-------------|-------------|
| `ethernet_socket` | RJ45 Ethernet | 13 × 9 mm |
| `power_socket` | IEC C13 kettle plug | 20 × 15 mm |
| `usb_port` | USB Type-A | 13 × 5 mm |
| `vga_socket` | VGA 15-pin | 35 × 12 mm |
| `hdmi_port` | HDMI Type-A | 14 × 5 mm |
| `audio_jack` | 3.5 mm audio | ⌀ 3.5 mm |
| `dvi_port` | DVI-D / DVI-I | 39 × 14 mm |
| `serial_port` | DE-9 serial (COM) | 20 × 9 mm |
| `ps2_port` | PS/2 keyboard/mouse | ⌀ 13 mm |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PORT_WORLD` | [0.270, 0.226, 0.835] m | VGA anchor world position |
| `CROP_PADDING` | 320 px | Half-size of crop around projected port |
| `IO_UPSCALE` | 8× and 12× | OWL-ViT upscale factors |
| `OWL_THRESHOLD` | 0.02 | Detection confidence threshold (max recall) |
| `NMS_IOU` | 0.40 | IoU threshold for cross-scale deduplication |
| `DEPTH_MIN` | 0.05 m | Minimum valid back-projection depth |
| `DEPTH_MAX` | 5.00 m | Maximum valid back-projection depth |
| `CAM_HEIGHT` | 0.83 m | Fallback metric anchor |
| `SKIP_FRAMES` | {400, 531} | Frames where IO panel is out of image |
| `nb_neighbors` | 30 | Open3D outlier removal neighbours |
| `std_ratio` | 1.5 | Open3D outlier removal σ multiplier |
| `min_inliers` | 10 | Min 3D points required to fit OBB |

---

## Troubleshooting

### No detections at all

1. Open `results/io_zoom_390.jpg` — do you see the IO panel ports?
   - **Yes** → lower `OWL_THRESHOLD` to `0.01` in `pipeline.py`
   - **No** → increase `CROP_PADDING` to `450` in `pipeline.py`

2. Check `results/preview_390.jpg` — is the cyan crosshair on the PC back panel?
   - **No** → the `PORT_WORLD` anchor may need adjusting. Measure a new anchor from the VGA GT.

### `ModuleNotFoundError: depth_anything_v2`

You forgot Step 4. Run:
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```
Make sure you clone it inside the `cp260/` project root (not inside `src/`).

### CUDA out of memory

Add before running:
```bash
export CUDA_VISIBLE_DEVICES=""   # force CPU
python src/pipeline.py
```
Or reduce `IO_UPSCALE` from `8.0` to `6.0` in `pipeline.py`.

### OBB not written for an entity

The entity had fewer than 10 inlier 3D points after outlier removal. Either:
- The port is not visible in enough frames
- Lower `OWL_THRESHOLD` further (try `0.01`)
- Lower `std_ratio` to `2.0` to keep more points

---

## Novel View Synthesis

3D Gaussian Splatting is used for the NVS deliverable:

```bash
# Clone 3DGS
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting

# Convert depth back-projections to initial point cloud (generated by pipeline.py)
# → use results/point_cloud_init.ply as initialisation

python train.py \
  -s ../data \
  --iterations 30000 \
  --densification_interval 100 \
  --opacity_reset_interval 3000 \
  --densify_until_iter 15000
```

Evaluation metrics: **PSNR**, **SSIM**, **LPIPS** on held-out views.

---

## Method Summary

```
Problem: 16 RGB frames, no depth, no CAD → OBBs for PC IO ports

Key insight 1:  Camera pose + known VGA anchor → exact per-frame crop, no detector.
Key insight 2:  Low threshold (0.02) + outlier removal >> high threshold + no noise.
Key insight 3:  Detect all 9 port types now → ready for any bonus entity on exam day.

Pipeline:
  pose geometry → crop → Depth Anything V2 → OWL-ViT (8×+12×) → NMS
  → back-project → aggregate 14 frames → outlier removal → OBB → answers.json
```

---

## References

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Yang et al., 2024
- [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) — Minderer et al., 2022
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Kerbl et al., 2023
- [Open3D](http://www.open3d.org/) — Zhou et al., 2018
