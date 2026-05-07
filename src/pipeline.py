import json, sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import pipeline as hf_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
FRAMES_DIR  = BASE_DIR / "data" / "frames"
POSES_FILE  = BASE_DIR / "data" / "poses.json"
RESULTS_DIR = BASE_DIR / "results"
DA2_DIR     = BASE_DIR / "Depth-Anything-V2"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Camera intrinsics ──
fx, fy = 1477.00974684544, 1480.4424455584467
cx, cy = 1298.2501500778505, 686.8201623541711
W_IMG, H_IMG = 2560, 1440
K      = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
K_inv  = np.linalg.inv(K)
CAM_HEIGHT = 0.83

# ── Known world position of the PC back-panel (from VGA ground truth) ──
# Used to compute a per-frame crop box — NO YOLO needed.
PORT_WORLD = np.array([0.2704921202927293, 0.2261220732082181, 0.8349008829378597])

# ── Crop padding around projected port location (pixels in full image) ──
# Increase if ports are near image edge and getting cut.
CROP_PADDING = 320   # was: IO panel fractions — now a fixed pixel pad

# ── OWL-ViT settings ──
IO_UPSCALE    = 8.0    # FIX: was 4.0 — higher = more pixels for tiny ports
OWL_THRESHOLD = 0.02   # FIX: was 0.05 — lower catches weak/small detections

# ── Labels: specific port types for the PC back panel ──
CANDIDATE_LABELS = [
    # Ethernet / RJ45
    "ethernet port", "rj45 port", "ethernet socket", "network port",
    "lan port", "rj45 socket", "network socket",
    # Power
    "power socket", "power connector", "kettle plug socket",
    "iec c13 socket", "electrical power port",
    # Other IO ports (helps OWL-ViT understand context)
    "usb port", "usb socket",
    "vga port", "hdmi port", "audio jack",
    "computer io panel", "pc back panel",
]

# ── Frames where the back panel is NOT visible — skip them ──
# Determined from pose analysis: port projects below image boundary
SKIP_FRAMES = {"400", "531"}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD POSES
# ─────────────────────────────────────────────────────────────────────────────
print("Loading poses...")
with open(POSES_FILE) as f:
    poses_raw = json.load(f)

imgs      = sorted([p.name for p in FRAMES_DIR.glob("frame_*.png")])
frame_ids = [str(int(n.replace("frame_","").replace(".png",""))) for n in imgs]

poses = {}
for fid in frame_ids:
    if fid in poses_raw:
        poses[fid] = np.array(poses_raw[fid], dtype=np.float64)
    else:
        print(f"  [WARN] No pose for frame {fid}")

print(f"  {len(poses)} poses loaded")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DEPTH ANYTHING V2
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading Depth Anything V2...")
sys.path.insert(0, str(DA2_DIR))
from depth_anything_v2.dpt import DepthAnythingV2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  device: {device}")

ckpt = hf_hub_download("depth-anything/Depth-Anything-V2-Small",
                        "depth_anything_v2_vits.pth")
depth_model = DepthAnythingV2(
    encoder="vits", features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(
    torch.load(ckpt, map_location="cpu", weights_only=False))
depth_model = depth_model.to(device).eval()
print("  Depth Anything V2 ready")

# NOTE: YOLO removed — we no longer need it.
# Tower detection was unreliable (detected whiteboard/wall instead of PC).
# We now use pose geometry to compute exact crop per frame.

# ─────────────────────────────────────────────────────────────────────────────
# LOAD OWL-ViT
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading OWL-ViT...")
owl = hf_pipeline(
    "zero-shot-object-detection",
    model="google/owlvit-base-patch32",
    device=0 if torch.cuda.is_available() else -1
)
print("  OWL-ViT ready")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_metric_depth(img_pil, pose_c2w):
    bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        rel = depth_model.infer_image(bgr)
    h, w  = rel.shape
    ch    = abs(pose_c2w[2, 3]) or CAM_HEIGHT
    scale = ch / (np.median(rel[h//3:2*h//3, w//3:2*w//3]) + 1e-6)
    return rel * scale


def pixel_to_world(u, v, depth_map, pose_c2w):
    """Back-project pixel → camera coords → world coords."""
    r = int(np.clip(v, 0, depth_map.shape[0]-1))
    c = int(np.clip(u, 0, depth_map.shape[1]-1))
    Z = float(depth_map[r, c])
    if not (0.05 < Z < 5.0):
        return None
    cam = K_inv @ np.array([u, v, 1.0]) * Z
    return (pose_c2w @ np.array([*cam, 1.0]))[:3]


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: POSE-BASED CROP  (replaces YOLO + IO fraction logic entirely)
# ─────────────────────────────────────────────────────────────────────────────
def get_port_crop_box(fid):
    """
    Project the known PORT_WORLD position into this frame using the camera pose.
    Returns (x1,y1,x2,y2) crop box, or None if port is not visible in this frame.

    WHY: YOLO was detecting a whiteboard/wall on the left side of the image
    instead of the actual PC tower on the right. Using pose geometry gives us
    an exact pixel location for every frame with zero detection errors.
    """
    pose_c2w = poses[fid]
    w2c = np.linalg.inv(pose_c2w)
    p_cam = w2c @ np.array([*PORT_WORLD, 1.0])

    if p_cam[2] <= 0:
        print(f"  [SKIP] Port is behind camera in frame {fid}")
        return None

    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy

    if not (0 < u < W_IMG and 0 < v < H_IMG):
        print(f"  [SKIP] Port projects outside image in frame {fid}: ({u:.0f},{v:.0f})")
        return None

    x1 = max(0,     int(u) - CROP_PADDING)
    x2 = min(W_IMG, int(u) + CROP_PADDING)
    y1 = max(0,     int(v) - CROP_PADDING)
    y2 = min(H_IMG, int(v) + CROP_PADDING)

    print(f"  Port projected to ({u:.0f},{v:.0f})  crop=({x1},{y1})→({x2},{y2})")
    return (x1, y1, x2, y2), (u, v)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: MULTI-SCALE OWL-ViT with NMS
# ─────────────────────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter  = interW * interH
    areaA  = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB  = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union  = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def nms(detections, iou_thresh=0.4):
    """Non-maximum suppression across all detections."""
    detections = sorted(detections, key=lambda d: d[1], reverse=True)
    kept = []
    for det in detections:
        _, conf, _, _, bbox = det
        suppress = False
        for _, _, _, _, kbbox in kept:
            if iou(bbox, kbbox) > iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept


def run_owlvit_on_crop(img_pil, crop_box_and_proj):
    """
    FIX 2: Run OWL-ViT at MULTIPLE scales and merge with NMS.
    Higher scales catch tiny ports that 4x missed.
    """
    crop_box, (proj_u, proj_v) = crop_box_and_proj
    cx1, cy1, cx2, cy2 = [int(v) for v in crop_box]
    crop = img_pil.crop((cx1, cy1, cx2, cy2))
    cW, cH = crop.size
    if cW < 5 or cH < 5:
        return []

    all_dets = []
    # FIX: Run at multiple scales instead of one fixed scale
    for scale_factor in [IO_UPSCALE, IO_UPSCALE * 1.5]:
        scale = min(scale_factor, 1600/cW, 1600/cH)
        big   = crop.resize((int(cW*scale), int(cH*scale)), Image.LANCZOS)
        try:
            raw = owl(big, candidate_labels=CANDIDATE_LABELS, threshold=OWL_THRESHOLD)
        except Exception as e:
            print(f"  [WARN] OWL-ViT error at scale {scale:.1f}x: {e}")
            continue

        for r in raw:
            b = r["box"]
            u_full = cx1 + (b["xmin"] + b["xmax"]) / 2 / scale
            v_full = cy1 + (b["ymin"] + b["ymax"]) / 2 / scale
            bbox_full = [
                cx1 + b["xmin"] / scale, cy1 + b["ymin"] / scale,
                cx1 + b["xmax"] / scale, cy1 + b["ymax"] / scale,
            ]
            all_dets.append((r["label"], float(r["score"]), u_full, v_full, bbox_full))

    # Deduplicate with NMS
    return nms(all_dets, iou_thresh=0.4)


def save_preview(img_pil, crop_box, proj_uv, detections, fid):
    """
    Cyan crosshair = projected port location (from pose).
    Yellow box     = crop region sent to OWL-ViT.
    Green boxes    = detected ports.
    Also saves io_zoom_<fid>.jpg.
    """
    img = np.array(img_pil.convert("RGB")).copy()

    # Draw crop box in yellow
    if crop_box:
        x1,y1,x2,y2 = [int(v) for v in crop_box]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,220,220), 3)
        cv2.putText(img, "port crop", (x1, max(y1-10,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,220,220), 2)

    # Draw projected port location as crosshair
    if proj_uv:
        pu, pv = int(proj_uv[0]), int(proj_uv[1])
        cv2.drawMarker(img, (pu, pv), (0,255,128),
                       cv2.MARKER_CROSS, 60, 4)
        cv2.putText(img, "projected port", (pu+10, pv-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,128), 2)

    # Draw detections in green
    for label, conf, u, v, bbox in detections:
        x1,y1,x2,y2 = [int(b) for b in bbox]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,220,50), 3)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1-8,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,220,50), 2)

    out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(RESULTS_DIR / f"preview_{fid}.jpg"), out_bgr)

    # IO zoom — what OWL-ViT actually sees
    if crop_box:
        ix1,iy1,ix2,iy2 = [int(v) for v in crop_box]
        crop_arr = out_bgr[iy1:iy2, ix1:ix2]
        if crop_arr.size > 0:
            zoom = cv2.resize(crop_arr, None, fx=3, fy=3,
                              interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(RESULTS_DIR / f"io_zoom_{fid}.jpg"), zoom)

    print(f"  Preview → results/preview_{fid}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE LOOP
# ─────────────────────────────────────────────────────────────────────────────
all_detections = []
print("\n── Running pipeline ──")

# FIX: Frames sorted by distance to port — process best frames first
# so you can QC results early without waiting for all 16 frames
FRAME_PRIORITY = ["390", "515", "371", "471", "365", "468",
                  "353", "496", "449", "426", "461", "359",
                  "333", "319", "400", "531"]
ordered_frame_ids = [f for f in FRAME_PRIORITY if f in frame_ids] + \
                    [f for f in frame_ids if f not in FRAME_PRIORITY]

for fid in ordered_frame_ids:
    if fid not in poses:
        continue

    # FIX: Skip frames where back panel is provably not visible
    if fid in SKIP_FRAMES:
        print(f"\nFrame {fid}: [SKIPPED — back panel not visible from this angle]")
        continue

    frame_path = FRAMES_DIR / f"frame_{int(fid):06d}.png"
    pose_c2w   = poses[fid]
    print(f"\nFrame {fid}:")

    img_pil = Image.open(frame_path)

    # FIX 1: Use pose geometry to get crop box (replaces YOLO)
    result = get_port_crop_box(fid)
    if result is None:
        continue
    crop_box_and_proj = result
    crop_box, proj_uv = result

    # Depth map (full image — needed for back-projection)
    depth_map = get_metric_depth(img_pil, pose_c2w)

    # FIX 2: Multi-scale OWL-ViT on the pose-derived crop
    dets = run_owlvit_on_crop(img_pil, crop_box_and_proj)
    print(f"  Detections after NMS: {len(dets)}")

    # Back-project to 3D
    frame_dets_3d = []
    for label, conf, u, v, bbox in dets:
        world = pixel_to_world(u, v, depth_map, pose_c2w)
        if world is None:
            continue
        all_detections.append({
            "frame_id":    fid,
            "class":       label,
            "confidence":  round(conf, 3),
            "bbox":        bbox,
            "pixel_center": [round(u, 1), round(v, 1)],
            "world_coords": world.tolist(),
        })
        frame_dets_3d.append((label, conf, u, v, bbox))
        print(f"  ✓ [{label}] conf={conf:.2f} px=({u:.0f},{v:.0f}) "
              f"world=({world[0]:.3f},{world[1]:.3f},{world[2]:.3f})")

    save_preview(img_pil, crop_box, proj_uv, frame_dets_3d, fid)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
out_file = RESULTS_DIR / "detections.json"
with open(out_file, "w") as f:
    json.dump(all_detections, f, indent=2)
print(f"\n✅ {len(all_detections)} detections → {out_file}")

# 3D scatter plot
if all_detections:
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    classes = list({d["class"] for d in all_detections})
    cmap    = dict(zip(classes, plt.cm.tab10(np.linspace(0, 1, len(classes)))))
    for d in all_detections:
        x, y, z = d["world_coords"]
        ax.scatter(x, y, z, color=cmap[d["class"]], s=80, label=d["class"])
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l,h)).values(), dict(zip(l,h)).keys())
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("PC Port Detections — 3D World Positions")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "3d_detections.png", dpi=150)
    plt.show()
    print("✅ 3D scatter plot saved.")
else:
    print("⚠️  No detections at all.")
    print("  → Check results/io_zoom_390.jpg — do you see the ports?")
    print("  → If yes: lower OWL_THRESHOLD further (try 0.01)")
    print("  → If no:  increase CROP_PADDING (try 400)")