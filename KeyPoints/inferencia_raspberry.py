#!/usr/bin/env python3
"""
inferencia_raspberry.py

Real-time person keypoint detection from webcam on Raspberry Pi 4 Model B.
Uses TorchVision's Keypoint R-CNN.

- Designed for low-memory, CPU-only environments (with CUDA warning).
- Efficient buffer management and reduced defaults for smooth Pi operation.
- All parameters configurable via CLI.
"""

import argparse
import logging
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

# ---- Additional imports for MobileNet, TorchScript, and advanced detection support ----
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection import KeypointRCNN
import os

def cpu_supports_fp16_matmul():
    """
    Return True if current torch CPU backend advertises fp16 matmul support.
    Safe on environments where torch.backends.cpu or matmul may be missing.
    """
    cpu_backend = getattr(torch.backends, "cpu", None)
    if cpu_backend is None:
        return False
    matmul_backend = getattr(cpu_backend, "matmul", None)
    if matmul_backend is None:
        return False
    return bool(getattr(matmul_backend, "allow_fp16", False))

# ==== DEFAULT CONFIGURATION FOR RASPBERRY PI ====
# These constants define the default optimization settings for best performance on Raspberry Pi 4.
# You can change them here or override on the CLI.
DEFAULT_CAP_WIDTH      = 320   # Camera capture width (pixels)
DEFAULT_CAP_HEIGHT     = 240   # Camera capture height (pixels)
DEFAULT_PROC_SCALE     = 0.5   # Fraction to downscale frames before inference (0.1-1.0)
DEFAULT_SKIP_FRAMES    = 2     # Number of frames to skip between inferences
DEFAULT_HALF_PRECISION = True  # Use float16 ("half precision") on supported hardware
TARGET_LATENCY_MS      = 200   # Target latency (ms) per inference, for adaptive skipping

# ---- Helper Functions ----

def parse_args():
    """Parse CLI arguments for runtime configuration."""
    parser = argparse.ArgumentParser(
        description="Real-time person keypoint detection with webcam on Raspberry Pi 4B."
    )
    parser.add_argument('--cam-idx', type=int, default=0,
                        help="Index of the webcam (default: 0)")
    parser.add_argument('--width', type=int, default=DEFAULT_CAP_WIDTH,
                        help=f"Capture width (default: {DEFAULT_CAP_WIDTH})")
    parser.add_argument('--height', type=int, default=DEFAULT_CAP_HEIGHT,
                        help=f"Capture height (default: {DEFAULT_CAP_HEIGHT})")
    parser.add_argument('--score-thr', type=float, default=0.8,
                        help="Score threshold for person detection (default: 0.8)")
    parser.add_argument('--draw-box', action='store_true',
                        help="Draw bounding boxes around detected persons")
    parser.add_argument('--log-file', type=str,
                        help="Optional log file to record events")
    parser.add_argument('--fps-window', type=int, default=30,
                        help="Number of frames for FPS smoothing (default: 30)")
    parser.add_argument('--no-window', action='store_true',
                        help="Disable OpenCV window display (useful for headless runs)")

    # === Optimization options ===
    parser.add_argument('--proc-scale', type=float, default=DEFAULT_PROC_SCALE,
                        help=f"Downscale factor for frame before inference (0.1-1.0, default: {DEFAULT_PROC_SCALE})")
    parser.add_argument('--skip-frames', type=int, default=DEFAULT_SKIP_FRAMES,
                        help=f"Number of frames to skip between inferences (default: {DEFAULT_SKIP_FRAMES} = every {DEFAULT_SKIP_FRAMES+1}th frame runs inference)")
    # Provide both --half and --no-half flags for user override (default from constant)
    half_group = parser.add_mutually_exclusive_group()
    half_group.add_argument('--half', dest='half', action='store_true',
                            help=f"Enable float16 (half precision) if supported (default: {'enabled' if DEFAULT_HALF_PRECISION else 'disabled'})")
    half_group.add_argument('--no-half', dest='half', action='store_false',
                            help=f"Disable float16 (half precision) (default: {'enabled' if DEFAULT_HALF_PRECISION else 'disabled'})")
    parser.set_defaults(half=DEFAULT_HALF_PRECISION)
    parser.add_argument('--target-latency-ms', type=int, default=TARGET_LATENCY_MS,
                        help=f"Soft latency budget per inference call in ms; adaptive frame skipping if not overridden (default: {TARGET_LATENCY_MS})")

    # === Model selection and acceleration ===
    parser.add_argument('--model-type', type=str, default='resnet50',
                        choices=['resnet50', 'mobile'],
                        help="Model backbone: 'resnet50' (default) or 'mobile' (MobilenetV3-Large+FPN, faster/lighter)")
    parser.add_argument('--torchscript', action='store_true',
                        help="EXPERIMENTAL: Run model as TorchScript for extra speed (fallback to eager if scripting fails)")
    return parser.parse_args()

def setup_logging(log_file=None):
    """
    Set up logging to console and optionally to file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=handlers
    )

def get_device():
    """
    Get the best available device (CUDA if present, else CPU).
    Warn if CUDA is unavailable (important for Pi).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        logging.warning("CUDA is NOT available. Running on CPU (expected on Raspberry Pi).")
    return device

def load_model(model_type: str, device, torchscript: bool):
    """
    Load a Keypoint R-CNN model according to the specified backbone and TorchScript option.
    Supported model_types:
        - 'resnet50': KeypointRCNN-ResNet50-FPN (pretrained COCO)
        - 'mobile': KeypointRCNN-MobileNetV3-Large-FPN (pretrained backbone, default RPN/ROI heads)
    If torchscript is True, attempt to load a cached scripted model or script it afresh.
    """
    cache_fname = f"keypointrcnn_{model_type}_scripted.pt"

    model = None
    if model_type == "resnet50":
        logging.info("Loading Keypoint R-CNN (ResNet50-FPN, pretrained on COCO)...")
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    elif model_type == "mobile":
        logging.info("Loading Keypoint R-CNN (MobileNetV3-Large-FPN backbone, pretrained)...")
        backbone = mobilenet_backbone('mobilenet_v3_large', pretrained=True, fpn=True)
        model = KeypointRCNN(
            backbone,
            num_classes=2,  # background + person
            num_keypoints=17,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    model.eval()

    # TorchScript logic
    if torchscript:
        cache_path = os.path.abspath(cache_fname)
        if os.path.exists(cache_path):
            try:
                logging.info(f"Loading TorchScript model from cache: {cache_fname}")
                scripted = torch.jit.load(cache_path, map_location=device)
                scripted.eval()
                return scripted
            except Exception as e:
                logging.warning(f"Failed to load cached TorchScript model: {e}. Falling back to eager model.")
        try:
            logging.info("Compiling model with torch.jit.script (experimental)...")
            scripted = torch.jit.script(model)
            scripted.eval()
            try:
                torch.jit.save(scripted, cache_path)
                logging.info(f"TorchScript model scripted and saved to: {cache_fname}")
            except Exception as e:
                logging.warning(f"TorchScript save failed (non-fatal): {e}")
            return scripted
        except Exception as e:
            logging.warning(f"TorchScript compilation failed: {e}. Using eager model.")
    logging.info("Model loaded and ready (eager mode).")
    return model

def get_keypoint_edges():
    """
    Returns the list of keypoint edges as pairs (from, to) for COCO skeleton.
    These define how to connect keypoints as lines for visualizing the person skeleton.
    """
    # COCO 17-keypoint skeleton edges (from COCO API/torchvision convention)
    return [
        (5, 7), (7, 9),      # left arm
        (6, 8), (8, 10),     # right arm
        (5, 6),              # shoulders
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16),  # right leg
        (11, 12),            # hips
        (5, 11), (6, 12),    # torso
        (1, 2), (0, 1), (0, 2),  # head
        (1, 3), (2, 4),      # eyes to ears
        (3, 5), (4, 6),      # ears to shoulders
    ]

def get_keypoint_colors():
    """
    Returns a list of BGR colors for drawing keypoints and skeleton edges.
    Colors are cycled for readability.
    """
    return [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (0, 0, 128),    # Maroon
        (128, 0, 0),    # Navy
        (0, 128, 0),    # Dark Green
        (0, 0, 0),      # Black
        (128, 128, 128),# Gray
        (0, 64, 255),   # Orange
        (255, 64, 0),   # Light Blue
        (64, 255, 0),   # Lime
    ]

def draw_keypoints_and_skeleton(img, keypoints, edges, kp_colors, edge_colors, draw_kp_radius=3, draw_edge_thick=2):
    """
    Draws keypoints and skeleton edges on the input image.

    Args:
        img: np.ndarray, image in BGR format (modified in-place).
        keypoints: np.ndarray, shape (N_kp, 3), where each row is (x, y, score).
        edges: list of (from, to) keypoint index pairs.
        kp_colors: list of BGR color tuples for keypoints.
        edge_colors: list of BGR color tuples for edges.
        draw_kp_radius: int, circle radius.
        draw_edge_thick: int, line thickness.
    """
    n_kp = keypoints.shape[0]
    for i in range(n_kp):
        x, y, score = keypoints[i]
        if score > 0.3:  # Draw only visible keypoints
            color = kp_colors[i % len(kp_colors)]
            cv2.circle(img, (int(x), int(y)), draw_kp_radius, color, -1, lineType=cv2.LINE_AA)

    for j, (start, end) in enumerate(edges):
        if (start < n_kp and end < n_kp and
                keypoints[start, 2] > 0.3 and keypoints[end, 2] > 0.3):
            color = edge_colors[j % len(edge_colors)]
            pt1 = tuple(map(int, keypoints[start, :2]))
            pt2 = tuple(map(int, keypoints[end, :2]))
            cv2.line(img, pt1, pt2, color, draw_edge_thick, lineType=cv2.LINE_AA)

def draw_bounding_box(img, bbox, color=(0, 255, 255), thickness=2):
    """
    Draw a bounding box on the image.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def compute_fps(deque_times):
    """
    Compute smoothed FPS from deque of recent inference times.
    """
    if len(deque_times) < 2:
        return 0.0
    avg_time = sum(deque_times) / len(deque_times)
    return 1.0 / avg_time if avg_time > 0 else 0.0

# -- Main Streaming Logic --

def main():
    """
    Main function: initializes camera, model, and runs frame-by-frame inference loop.
    Implements frame downscaling, adaptive frame skipping, and half-precision inference for optimized Raspberry Pi performance.
    """
    args = parse_args()
    setup_logging(args.log_file)

    logging.info(
        f"Starting webcam keypoint detection (camera idx={args.cam_idx}, res={args.width}x{args.height})"
    )
    logging.info(
        f"proc_scale={args.proc_scale} | skip_frames={args.skip_frames} | half_precision={'ON' if args.half else 'OFF'} | "
        f"target_latency_ms={getattr(args, 'target_latency_ms', TARGET_LATENCY_MS)} | "
        f"model_type={args.model_type} | torchscript={'ENABLED' if args.torchscript else 'OFF'}"
    )

    # Select device
    device = get_device()

    # Load model with new args
    model = load_model(args.model_type, device, args.torchscript)

    # === Attempt to enable half-precision inference if requested ===
    use_half = False
    if args.half:
        half_supported = False
        if device.type == "cuda":
            half_supported = True
        elif device.type == "cpu":
            half_supported = cpu_supports_fp16_matmul()
        logging.info(f"CPU fp16 matmul support detected: {half_supported}")
        if half_supported:
            try:
                model.half()
                use_half = True
                logging.info("Model set to half precision (float16).")
            except Exception as e:
                logging.warning(f"Failed to set model to half precision: {e}")
        else:
            logging.warning("--half requested but not supported on this device. Running in float32.")

    # Validate proc_scale
    if not (0.1 <= args.proc_scale <= 1.0):
        logging.warning(f"Invalid --proc-scale: {args.proc_scale}, resetting to 1.0")
        args.proc_scale = 1.0

    # Get skeleton edges and colors
    edges = get_keypoint_edges()
    kp_colors = get_keypoint_colors()
    edge_colors = get_keypoint_colors()  # Reuse for edges, cycle if needed

    # Set up FPS smoothing (counts displayed frames, not just inference frames)
    frame_times = deque(maxlen=args.fps_window)

    # Open camera
    cap = cv2.VideoCapture(args.cam_idx)
    if not cap.isOpened():
        logging.error(f"Could not open camera index {args.cam_idx}")
        sys.exit(1)
    # Set capture resolution (important for Pi performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win_name = "Keypoint Detection (press 'q' to quit)"

    # === Frame skipping and prediction cache ===
    # The skip_frames variable may be adapted at runtime to honor the target latency budget.
    skip_frames = max(0, args.skip_frames)
    frame_counter = 0
    last_preds = None
    last_boxes, last_labels, last_scores, last_keypoints, last_person_indices = [], [], [], [], []

    # Determine if user has overridden skip_frames (if not, enable adaptive logic)
    user_override_skip = (args.skip_frames != DEFAULT_SKIP_FRAMES)
    target_latency_ms = getattr(args, "target_latency_ms", TARGET_LATENCY_MS)

    try:
        while True:
            # 1. Capture frame
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame from camera.")
                break

            orig_h, orig_w = frame.shape[:2]

            # 2. Downscale workflow: compute scaled frame for inference
            if args.proc_scale < 1.0:
                scaled_w = max(1, int(orig_w * args.proc_scale))
                scaled_h = max(1, int(orig_h * args.proc_scale))
                scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            else:
                scaled_frame = frame.copy()
                scaled_w, scaled_h = orig_w, orig_h

            # 3. Preprocess: Convert BGR->RGB, to torch tensor, normalize [0,1]
            rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            input_tensor = F.to_tensor(rgb).to(device)
            if use_half:
                try:
                    input_tensor = input_tensor.half()
                except Exception as e:
                    logging.warning(f"Failed to convert input tensor to half: {e}")
                    input_tensor = input_tensor.float()

            # 4. Frame skipping workflow
            need_infer = (frame_counter % (skip_frames + 1) == 0)
            inf_time = 0.0
            if need_infer or last_preds is None:
                # --- Run model inference ---
                t0 = time.perf_counter()
                with torch.no_grad():
                    preds = model([input_tensor])[0]
                t1 = time.perf_counter()
                inf_time = t1 - t0
                last_preds = preds
                # Parse predictions & cache for skipped frames
                last_boxes = preds['boxes'].cpu().numpy() if len(preds['boxes']) > 0 else []
                last_labels = preds['labels'].cpu().numpy() if len(preds['labels']) > 0 else []
                last_scores = preds['scores'].cpu().numpy() if len(preds['scores']) > 0 else []
                last_keypoints = preds['keypoints'].cpu().numpy() if len(preds['keypoints']) > 0 else []
                last_person_indices = [
                    i for i, (label, score) in enumerate(zip(last_labels, last_scores))
                    if label == 1 and score >= args.score_thr
                ]

                # === Dynamic frame-skip adjustment (adaptive skip_frames) ===
                # Only auto-adjust if the user did not override skip_frames at CLI.
                if not user_override_skip and target_latency_ms > 0:
                    new_skip = max(0, int((inf_time * 1000) / target_latency_ms) - 1)
                    if new_skip != skip_frames:
                        logging.info(f"Auto-adjust skip_frames {skip_frames} → {new_skip} to honor latency budget ({target_latency_ms} ms)")
                        skip_frames = new_skip

            # -- Rescale predictions to original frame coordinates if downscaled --
            scale_x = orig_w / scaled_w
            scale_y = orig_h / scaled_h

            # Draw all detected persons (use cached or fresh predictions)
            for idx in last_person_indices:
                kp = last_keypoints[idx]  # shape (17, 3)
                kp_scaled = kp.copy()
                kp_scaled[:, 0] *= scale_x
                kp_scaled[:, 1] *= scale_y
                draw_keypoints_and_skeleton(
                    frame, kp_scaled, edges, kp_colors, edge_colors
                )
                if args.draw_box:
                    # Draw bounding box if requested (rescale coords)
                    bbox = last_boxes[idx]
                    bbox_scaled = [
                        bbox[0] * scale_x, bbox[1] * scale_y,
                        bbox[2] * scale_x, bbox[3] * scale_y
                    ]
                    draw_bounding_box(frame, bbox_scaled)

            # 5. Overlay FPS (based on total displayed frames, not just inference calls)
            t_display = time.perf_counter()
            frame_times.append(t_display)
            if len(frame_times) > 1:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0] + 1e-8)
            else:
                fps = 0.0
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, lineType=cv2.LINE_AA
            )
            if skip_frames > 0:
                cv2.putText(
                    frame, f"(skip {skip_frames})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame, f"(skip {skip_frames})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, lineType=cv2.LINE_AA
                )

            # 6. Display frame (unless --no-window)
            if not args.no_window:
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Quit signal received ('q' pressed). Exiting.")
                    break

            # 7. Log timing and inference only when model was run
            if need_infer or frame_counter == 0:
                logging.info(
                    f"Inference time: {inf_time*1000:.1f} ms | FPS: {fps:.2f} | Persons: {len(last_person_indices)}"
                )
                # Print parseable line to stdout as required
                print(f"Inference time: {inf_time*1000:.1f} ms | FPS: {fps:.2f} | Persons: {len(last_person_indices)}")
            frame_counter += 1

    except KeyboardInterrupt:
        logging.info("Interrupted by user (Ctrl+C). Exiting.")

    finally:
        # Clean up
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        logging.info("Camera released and all windows closed. Goodbye!")

if __name__ == "__main__":
    main()