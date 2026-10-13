"""
Image Grid Layout Script
========================
Arranges images side by side with labels and saves as a single .png

Usage:
    # Original style: list each image explicitly
    python image_grid.py img1.png img2.png img3.png

    # New style: point at a folder
    python image_grid.py --input-dir /path/to/segmented
    python image_grid.py --input-dir /path/to/segmented --pattern "bacteria*.png"
    python image_grid.py --input-dir /path/to/segmented --pattern "bacteria*.png" \
                         --timepoints 0 5 10 15 20

You can mix flags freely. Labels default to "<N> hr" parsed from filenames
like bacteria.png; pass --labels to override.

More than MAX_COLS images will automatically wrap into multiple rows.
"""

import argparse
import glob
import math
import os
import re

import cv2
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================
LABEL_FONTSIZE  = 18
FIGURE_HEIGHT   = 4     # inches per row
DPI             = 150
MAX_COLS        = 5     # images per row before wrapping

CONTROL_LABEL_FONTSIZE = 12

# Well grid dimensions (must match well_detector.py)
N_ROWS = 5
N_COLS = 11

# Resize images to this width before Hough detection (same as well_detector.py default)
DETECTION_MAX_WIDTH = 1000

# HoughCircles — tuned to match well_detector.py for ~1000 px wide images
HOUGH_DP         = 1.2
HOUGH_MIN_DIST   = 42
HOUGH_PARAM1     = 50
HOUGH_PARAM2     = 25
HOUGH_MIN_RADIUS = 16
HOUGH_MAX_RADIUS = 30

# Fallback box as image fractions if all detection fails (left, top, w, h)
CONTROL_BOX_FALLBACK = (0.03, 0.08, 0.08, 0.80)
# =============================================================================


def parse_timepoint(filename: str):
    """Extract integer timepoint from a name like 'bacteria.png' -> 5."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"_(\d+)$", stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Well-detection helpers (ported from well_detector.py)
# ---------------------------------------------------------------------------

def _hough_detect(image_bgr):
    """Detect circles and filter obvious border false-positives."""
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int).tolist()
    h, w = image_bgr.shape[:2]
    margin      = HOUGH_MAX_RADIUS
    margin_bot  = HOUGH_MAX_RADIUS * 3   # extra tight at the bottom (device rim)
    return [
        c for c in circles
        if margin < c[0] < w - margin
        and c[1] > margin
        and c[1] < h - margin_bot
    ]


def _cluster_values(vals, gap=30):
    """Cluster sorted 1-D values; return group means."""
    vals   = np.sort(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [np.mean(g) for g in groups]


def _fit_axis(positions, expected_count):
    """
    Fit a regular grid along one axis.  Handles missing rows/cols at either
    edge by inferring true indices before regression (same logic as well_detector.py).
    Returns `expected_count` evenly-spaced positions.
    """
    positions = np.array(positions)
    spacing   = np.median(np.diff(positions)) if len(positions) > 1 else positions[0] * 0.15
    raw_idx   = np.round((positions - positions[0]) / spacing).astype(int)
    slope, intercept = np.polyfit(raw_idx, positions, 1)
    max_start = raw_idx[-1] - expected_count + 1
    start     = max(0, min(raw_idx[0], max_start))
    idx       = np.arange(start, start + expected_count)
    return slope * idx + intercept


def _fit_grid(detected):
    """Return (wells, scale_factor) where wells is a list of (x,y,r) pixel
    tuples for every grid position, and scale_factor converts those coords
    back to the original (unresized) image."""
    centers  = np.array([(x, y) for x, y, r in detected])
    median_r = int(np.median([r for x, y, r in detected]))

    col_raw = np.array(_cluster_values(centers[:, 0], gap=30))
    row_raw = np.array(_cluster_values(centers[:, 1], gap=30))

    col_centers = _fit_axis(col_raw, N_COLS)
    row_centers = _fit_axis(row_raw, N_ROWS)

    wells = [
        (int(round(cx)), int(round(ry)), median_r)
        for ry in row_centers
        for cx in col_centers
    ]
    return wells


def _control_box_pixels(wells, pad=10):
    """
    Return (x, y, w, h) pixel bounding box around the leftmost (control)
    column of wells — identical logic to _control_column_rect in well_detector.py.
    """
    col0 = [w for i, w in enumerate(wells) if i % N_COLS == 0]
    if not col0:
        return None
    xs = [w[0] for w in col0]
    ys = [w[1] for w in col0]
    rs = [w[2] for w in col0]
    p  = max(rs) + pad
    x  = min(xs) - p
    y  = min(ys) - p
    bw = (max(xs) - min(xs)) + 2 * p
    bh = (max(ys) - min(ys)) + 2 * p
    return (x, y, bw, bh)


def detect_control_box_pixels(img_path: str):
    """
    Full pipeline: resize → Hough → grid-fit → leftmost-column bounding box.

    Returns (x, y, w, h) in the *original* image's pixel space,
    or None if detection fails.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None

    orig_h, orig_w = bgr.shape[:2]

    # Resize to the width the Hough params were tuned for
    if orig_w > DETECTION_MAX_WIDTH:
        scale  = DETECTION_MAX_WIDTH / orig_w
        bgr    = cv2.resize(bgr, (DETECTION_MAX_WIDTH, int(orig_h * scale)),
                            interpolation=cv2.INTER_AREA)
    else:
        scale  = 1.0

    detected = _hough_detect(bgr)
    print(f"  [control box] Hough detected {len(detected)} circles")

    if len(detected) < max(N_ROWS, N_COLS):
        print("  [control box] Too few circles for grid fit — using fallback.")
        return None

    try:
        wells = _fit_grid(detected)
    except Exception as e:
        print(f"  [control box] Grid fit failed ({e}) — using fallback.")
        return None

    box = _control_box_pixels(wells, pad=10)
    if box is None:
        return None

    # Scale coords back to original image size
    x, y, bw, bh = box
    inv = 1.0 / scale
    box_orig = (x * inv, y * inv, bw * inv, bh * inv)
    print(f"  [control box] Control column: x={box_orig[0]:.0f}, y={box_orig[1]:.0f}, "
          f"w={box_orig[2]:.0f}, h={box_orig[3]:.0f}")
    return box_orig


def draw_control_box(ax, img_shape, box):
    """
    Draw the control rectangle and label onto *ax*.

    *box* is either:
      - a 4-tuple of pixel coords  (x, y, w, h)          from detect_control_box_pixels
      - a 4-tuple of image fractions (left, top, fw, fh)  from CONTROL_BOX_FALLBACK
      The two cases are distinguished by whether values are > 1 (pixels) or ≤ 1 (fractions).
    """
    h, w = img_shape[:2]
    x, y, bw, bh = box

    # Convert fractions → pixels if needed
    if max(bw, bh) <= 1.0:
        x, bw = x * w, bw * w
        y, bh = y * h, bh * h

    rect = mpatches.Rectangle(
        (x, y), bw, bh,
        linewidth=2.5,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(rect)

    ax.text(
        x + bw / 2, y - h * 0.03, "Control",
        fontsize=CONTROL_LABEL_FONTSIZE, ha="center", va="bottom",
        color="black", fontweight="bold",
    )


def save_image_grid(image_paths, labels, out_path):
    n       = len(image_paths)
    n_cols  = min(n, MAX_COLS)
    n_rows  = math.ceil(n / MAX_COLS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, FIGURE_HEIGHT * n_rows),
    )

    # Normalise axes to a 2-D list for uniform indexing
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]

    # Detect the control box independently per image to follow drift between
    # timepoints.  Falls back to the most recent successful detection, or the
    # hardcoded fractions if nothing has worked yet.
    last_good_box = None

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        row, col = divmod(idx, MAX_COLS)
        ax = axes[row][col]

        if not os.path.exists(img_path):
            print(f"WARNING: File not found, skipping: {img_path}")
            ax.axis("off")
            continue

        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(label, fontsize=LABEL_FONTSIZE, pad=8)
        ax.axis("off")

        print(f"\nDetecting control column from: {os.path.basename(img_path)}")
        box = detect_control_box_pixels(img_path)

        if box is not None:
            last_good_box = box
        elif last_good_box is not None:
            print("  Detection failed — reusing previous timepoint's box.")
            box = last_good_box
        else:
            print("  Detection failed — using hardcoded fallback box.")
            box = CONTROL_BOX_FALLBACK

        draw_control_box(ax, img.shape, box)

    # Hide any unused axes in the last row
    total_cells = n_rows * n_cols
    for extra in range(n, total_cells):
        row, col = divmod(extra, MAX_COLS)
        axes[row][col].axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.15,
                        top=0.92, bottom=0.02, left=0.02, right=0.98)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def collect_images(input_dir, pattern, timepoints):
    """Return a sorted list of image paths from input_dir matching pattern,
    optionally filtered by --timepoints."""
    full_pattern = os.path.join(input_dir, pattern)
    paths = sorted(glob.glob(full_pattern))
    if not paths:
        raise SystemExit(f"No images matched: {full_pattern}")

    if timepoints is not None:
        wanted   = set(timepoints)
        filtered = [p for p in paths if parse_timepoint(p) in wanted]
        found    = {parse_timepoint(p) for p in filtered}
        missing  = sorted(wanted - found)
        if missing:
            print(f"WARNING: requested timepoints not found: {missing}")
        if not filtered:
            raise SystemExit(f"No images at requested timepoints {sorted(wanted)}")
        print(f"Filtered {len(paths)} -> {len(filtered)} images "
              f"at timepoints {sorted(found)}")
        return filtered

    print(f"Found {len(paths)} images matching {pattern} in {input_dir}")
    return paths


def default_label(path):
    """Build a friendly default label like '5 hr' from bacteria.png."""
    tp = parse_timepoint(path)
    if tp is not None:
        return f"{tp} hr"
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arrange images into a labeled grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", nargs="*",
                        help="Image paths. Required if --input-dir is not used.")
    parser.add_argument("--input-dir",
                        help="Folder of images to pull from. When set, the "
                             "positional 'images' argument is ignored.")
    parser.add_argument("--pattern", default="*.png",
                        help="Glob pattern within --input-dir (e.g. 'bacteria*.png').")
    parser.add_argument("--timepoints", type=int, nargs="+", default=None,
                        metavar="T",
                        help="Only include specific timepoints parsed from "
                             "filenames like bacteria.png (e.g. --timepoints 0 5 10).")
    parser.add_argument("--labels", nargs="+",
                        help="Override labels for each image. If omitted, "
                             "labels default to '<N> hr' parsed from filenames.")
    parser.add_argument("--output",
                        help="Output path. If omitted, derived from the first "
                             "image's prefix.")
    parser.add_argument("--max-cols", type=int, default=MAX_COLS,
                        help="Maximum images per row before wrapping.")
    args = parser.parse_args()

    # Allow overriding MAX_COLS via CLI
    if args.max_cols != MAX_COLS:
        MAX_COLS = args.max_cols

    if args.input_dir:
        images = collect_images(args.input_dir, args.pattern, args.timepoints)
    elif args.images:
        images = args.images
    else:
        parser.error("Provide image paths positionally or use --input-dir.")

    if args.labels:
        if len(args.labels) != len(images):
            parser.error(f"--labels has {len(args.labels)} entries but "
                         f"{len(images)} images were selected.")
        labels = args.labels
    else:
        labels = [default_label(p) for p in images]

    if args.output:
        out_path = args.output
    else:
        prefix = os.path.splitext(os.path.basename(images[0]))[0].rsplit("_", 1)[0]
        out_path = os.path.join("Timelapse Arrangement", prefix, prefix + "_grid.png")

    save_image_grid(images, labels, out_path)