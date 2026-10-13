"""
Well Detection Script
=====================
Detects and classifies positive (yellow-orange) vs negative (white-grey) wells
in a 55-well (5 rows x 11 cols) device image.

Requirements:
    pip install opencv-python scikit-image matplotlib scipy numpy

Usage:
    python well_detector.py <input_dir> <output_dir> <bacteria> [--composite] [--show]

    Analyzes all files matching <bacteria>_*.png inside <input_dir> and writes
    results into <output_dir>/<bacteria>/.

Examples:
    python well_detector.py /data/segmented /data/results bacteria
    python well_detector.py /data/segmented /data/results bacteria --composite
    python well_detector.py /data/segmented /data/results bacteria --show
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import glob
import os
import re
import sys
import argparse


# =============================================================================
# CONFIGURATION — Adjust these parameters if needed
# =============================================================================

# HSV color thresholds for yellow-orange positive wells
# Derived from Fiji HSB calibration 
LOWER_YELLOW = np.array([14,  45,  175])   # [H_min, S_min, V_min]
UPPER_YELLOW = np.array([35, 255, 235])   # [H_max, S_max, V_max]

# Well grid dimensions
N_ROWS = 5
N_COLS = 11
N_WELLS = N_ROWS * N_COLS  # 55

# Hough circle detection parameters
HOUGH_DP         = 1.2   # Inverse resolution ratio
HOUGH_MIN_DIST   = 42    # Minimum distance between circle centers (px)
HOUGH_PARAM1     = 50    # Canny edge threshold
HOUGH_PARAM2     = 25    # Accumulator threshold (lower = more circles detected)
HOUGH_MIN_RADIUS = 16    # Minimum well radius (px)
HOUGH_MAX_RADIUS = 30    # Maximum well radius (px)

# Classification threshold — fraction of pixels in well that must be yellow to call positive
POSITIVE_THRESHOLD = 0.25  # 25%

# Grid fitting — max distance (px) from grid point to accept a detected circle as a match
GRID_MATCH_DIST = 25

# Morphological cleanup kernel size for color mask
MORPH_KERNEL_SIZE = 5

# Output DPI for saved figure
OUTPUT_DPI = 150

# Scale factor for well circle radius (1.0 = use median detected radius as-is)
RADIUS_SCALE = 1.0


# =============================================================================


def load_image(path, max_width=None):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if max_width is not None and image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        new_size = (max_width, int(round(image.shape[0] * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image, image_rgb, image_hsv


def build_color_mask(image_hsv):
    """Create and clean up the yellow-orange color mask."""
    mask = cv2.inRange(image_hsv, LOWER_YELLOW, UPPER_YELLOW)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_circles_hough(image_bgr):
    """Detect well circles using Hough Circle Transform."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
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
    
    circles = np.round(circles[0, :]).astype("int").tolist()
    
    # Filter out circles too close to the image border
    h, w = image_bgr.shape[:2]
    margin_side = HOUGH_MAX_RADIUS      # left, right, top — just enough to catch obvious false circles
    margin_bottom = HOUGH_MAX_RADIUS * 3  # extra tight on bottom to block device rim

    circles = [
        c for c in circles
        if margin_side < c[0] < w - margin_side      # left/right
        and c[1] > margin_side                        # top
        and c[1] < h - margin_bottom                  # bottom (stricter)
    ]
    
    return circles


def cluster_values(vals, gap_threshold=30):
    """Cluster 1D values into groups and return group means (used for grid fitting)."""
    vals = np.sort(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] > gap_threshold:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [np.mean(g) for g in groups]


def fit_grid(detected_circles):
    """
    Fit a regular N_ROWS x N_COLS grid to detected circles using linear regression.
    Correctly handles missing rows/columns at either edge by inferring true grid indices
    before fitting rather than assuming detections start at index 0.
    """
    centers = np.array([(x, y) for x, y, r in detected_circles])
    median_r = int(np.median([r for x, y, r in detected_circles]))

    col_centers_raw = np.array(cluster_values(centers[:, 0], gap_threshold=30))
    row_centers_raw = np.array(cluster_values(centers[:, 1], gap_threshold=30))

    print(f"  Grid: {len(row_centers_raw)} rows x {len(col_centers_raw)} cols detected")

    def fit_with_true_indices(positions, expected_count):
        """
        Estimate the true grid index of each detected position, then fit a line
        through (true_index, position) pairs and extrapolate to all expected_count slots.
        This correctly handles missing rows/cols at the start OR end.
        """
        positions = np.array(positions)
        if len(positions) == 1:
            # Can't fit a line with one point — use a reasonable default spacing
            spacing = positions[0] * 0.15
        else:
            spacing = np.median(np.diff(positions))

        # Assign each detected position its nearest integer grid index
        # anchored from the first detected position
        raw_indices = np.round((positions - positions[0]) / spacing).astype(int)

        # Fit line through (raw_index, position) pairs
        slope, intercept = np.polyfit(raw_indices, positions, 1)

        # Now find which contiguous block of expected_count indices best covers
        # the detected ones — i.e. find the offset so we don't always start at 0
        # The detected indices span raw_indices[0]..raw_indices[-1]
        # We need to fit them into a window of size expected_count
        max_start = raw_indices[-1] - expected_count + 1
        start = max(0, min(raw_indices[0], max_start))

        all_indices = np.arange(start, start + expected_count)
        return slope * all_indices + intercept

    col_centers = fit_with_true_indices(col_centers_raw, N_COLS)
    row_centers = fit_with_true_indices(row_centers_raw, N_ROWS)

    print(f"  Fitted col centers: {[round(c) for c in col_centers]}")
    print(f"  Fitted row centers: {[round(r) for r in row_centers]}")

    # Use fitted grid position and median radius for every well
    final_wells = []
    for ry in row_centers:
        for cx in col_centers:
            final_wells.append((int(round(cx)), int(round(ry)), int(median_r * RADIUS_SCALE)))

    print(f"  Total wells: {len(final_wells)}")
    return final_wells


def classify_wells(final_wells, image_hsv, mask_clean):
    """Classify each well as positive or negative based on overlap with pre-built mask."""
    results = []
    for (x, y, r) in final_wells:
        r_inner = int(r * 0.8)
        y1 = max(0, y - r_inner)
        y2 = min(image_hsv.shape[0], y + r_inner)
        x1 = max(0, x - r_inner)
        x2 = min(image_hsv.shape[1], x + r_inner)

        # Use the pre-built clean mask instead of re-thresholding
        roi_mask = mask_clean[y1:y2, x1:x2]

        # Circular mask — offset center to account for edge clipping
        actual_cy = y - y1
        actual_cx = x - x1
        Y, X = np.ogrid[:roi_mask.shape[0], :roi_mask.shape[1]]
        circle_mask = ((X - actual_cx)**2 + (Y - actual_cy)**2) <= r_inner**2

        yellow_px = np.sum((roi_mask > 0) & circle_mask)
        total_px  = np.sum(circle_mask)
        fraction  = yellow_px / total_px if total_px > 0 else 0.0

        results.append({
            "center":   (x, y),
            "radius":   r,
            "fraction": fraction,
            "positive": fraction >= POSITIVE_THRESHOLD,
        })
    return results


def save_csv(results, out_path):
    """Save results to CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["well_number", "x_center", "y_center", "yellow_fraction", "result"])
        for i, r in enumerate(results):
            writer.writerow([
                i + 1,
                r["center"][0],
                r["center"][1],
                f"{r['fraction']:.3f}",
                "POSITIVE" if r["positive"] else "NEGATIVE",
            ])
    print(f"  CSV saved: {out_path}")


def _control_column_rect(results, pad):
    """
    Build a single mpatches.Rectangle that encloses every well in the
    leftmost (control) column. Returns None if there are no controls.
    """
    control_wells = [r for i, r in enumerate(results) if i % N_COLS == 0]
    if not control_wells:
        return None
    xs = [w["center"][0] for w in control_wells]
    ys = [w["center"][1] for w in control_wells]
    rs = [w["radius"]    for w in control_wells]
    p = max(rs) + pad
    return mpatches.Rectangle(
        (min(xs) - p, min(ys) - p),
        (max(xs) - min(xs)) + 2 * p,
        (max(ys) - min(ys)) + 2 * p,
        fill=False, edgecolor="black", linewidth=2.5,
    )


def save_figure(image_rgb, mask_clean, results, out_path, show=False):
    n_pos   = sum(1 for r in results if r["positive"])
    n_total = len(results)
    n_ctrl  = N_ROWS
    n_neg   = n_total - n_pos - n_ctrl

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(mask_clean, cmap="hot")
    axes[1].set_title("Yellow-Orange Mask", fontsize=12)
    axes[1].axis("off")
    axes[2].imshow(image_rgb)
    ax = axes[2]

    for i, r in enumerate(results):
        x, y  = r["center"]
        color = "orangered" if r["positive"] else "deepskyblue"
        inner = plt.Circle((x, y), r["radius"] + 3, color=color, fill=False, linewidth=2.5)
        ax.add_patch(inner)

    # Single black rectangle around the entire control column (replaces per-well circles)
    rect = _control_column_rect(results, pad=10)
    if rect is not None:
        ax.add_patch(rect)

    pos_patch  = mpatches.Patch(color="orangered", label=f"Positive ({n_pos})")
    neg_patch  = mpatches.Patch(color="deepskyblue",  label=f"Negative ({n_neg})")
    ctrl_patch = mpatches.Patch(color="black",        label=f"Control ({n_ctrl})")

    fig.subplots_adjust(right=0.85, wspace=0.05)

    legend_ax = fig.add_axes([0.86, 0.50, 0.10, 0.15])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[pos_patch, neg_patch, ctrl_patch],
        loc="upper left",
        fontsize=12,
        frameon=True,
    )

    ax.set_title(f"{n_pos} positive  /  {n_total} total wells", fontsize=12)
    ax.axis("off")

    plt.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    print(f"  Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close()


def save_composite_figure(image_results, out_path, show=False):
    n_images = len(image_results)
    fig, axes = plt.subplots(n_images, 2, figsize=(18, 3.5 * n_images))

    if n_images == 1:
        axes = [axes]

    # Lowered `top` from 0.97 to 0.93 to leave room for the column headers above.
    fig.subplots_adjust(right=0.77, wspace=0.05, hspace=0.02, top=0.93, left=0.05)

    for row, (timepoint_label, image_rgb, results) in enumerate(image_results):
        n_pos   = sum(1 for r in results if r["positive"])

        axes[row][0].imshow(image_rgb)
        axes[row][0].axis("off")
        axes[row][1].imshow(image_rgb)
        axes[row][1].axis("off")

        for i, r in enumerate(results):
            x, y  = r["center"]
            color = "orangered" if r["positive"] else "deepskyblue"
            inner = plt.Circle((x, y), r["radius"] + 3, color=color, fill=False, linewidth=2.5)
            axes[row][1].add_patch(inner)

        # Single black rectangle around the entire control column
        rect = _control_column_rect(results, pad=8)
        if rect is not None:
            axes[row][1].add_patch(rect)

        # Get exact vertical center of this row from the axis position
        pos = axes[row][0].get_position()
        row_center = (pos.y0 + pos.y1) / 2

        # Positive fraction on the right, rotated 90 degrees
        fig.text(
            0.95, row_center - 0.06, "Positive Droplets: " f"{n_pos}",
            fontsize=22, va="center", ha="right",
            fontweight="bold"
        )

    # Column headers above the two image columns
    if n_images > 0:
        left_pos  = axes[0][0].get_position()
        right_pos = axes[0][1].get_position()
        left_x    = (left_pos.x0  + left_pos.x1)  / 2
        right_x   = (right_pos.x0 + right_pos.x1) / 2
        header_y  = max(left_pos.y1, right_pos.y1) + 0.015

        fig.text(left_x,  header_y, "Raw Images",
                 fontsize=28, fontweight="bold", ha="center", va="bottom")
        fig.text(right_x, header_y, "Analyzed Images",
                 fontsize=28, fontweight="bold", ha="center", va="bottom")

    # Legend
    pos_patch  = mpatches.Patch(color="orangered", label="Positive")
    neg_patch  = mpatches.Patch(color="deepskyblue",  label="Negative")
    ctrl_patch = mpatches.Patch(color="black",        label="Control")
    fig.legend(
        handles=[pos_patch, neg_patch, ctrl_patch],
        loc="right",
        bbox_to_anchor=(0.94, 0.905),
        fontsize=28,
        frameon=True,
        edgecolor="lightgrey",
    )

    plt.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    print(f"  Composite figure saved: {out_path}")
    if show:
        plt.show()
    plt.close()


def print_summary(results):
    """Print a summary table to stdout."""
    n_pos = sum(1 for r in results if r["positive"])
    print(f"\n{'='*50}")
    print(f"  RESULTS: {n_pos} POSITIVE / {len(results)} TOTAL WELLS")
    print(f"{'='*50}")
    print(f"  {'Well':>5}  {'Center':^14}  {'Yellow%':>7}  {'Result'}")
    print(f"  {'-'*45}")
    for i, r in enumerate(results):
        status = "POSITIVE ✓" if r["positive"] else "negative"
        print(f"  {i+1:>5}  {str(r['center']):^14}  {r['fraction']:>6.1%}   {status}")
    print(f"{'='*50}\n")


def run(image_path, output_dir, bacteria, max_width=None,
        show=False, composite_mode=False):
    base        = os.path.splitext(os.path.basename(image_path))[0]
    out_dir     = os.path.join(output_dir, bacteria, "Three_Panel")
    os.makedirs(out_dir, exist_ok=True)
    out_png     = os.path.join(out_dir, base + "_results.png")
    out_csv     = os.path.join(out_dir, base + "_results.csv")

    print(f"\nLoading image: {image_path}")
    image, image_rgb, image_hsv = load_image(image_path, max_width=max_width)
    print(f"  Image size: {image.shape[1]} x {image.shape[0]} px")

    print("\nBuilding color mask...")
    mask_clean = build_color_mask(image_hsv)

    print("\nDetecting wells (Hough circles)...")
    detected = detect_circles_hough(image)

    print(f"  Hough detected: {len(detected)} circles")

    print("\nFitting grid to find all 55 wells...")
    final_wells = fit_grid(detected)
    print(f"  Total wells: {len(final_wells)}")

    print("\nClassifying wells...")
    results = classify_wells(final_wells, image_hsv, mask_clean)

    print_summary(results)

    print("Saving outputs...")
    if not composite_mode:
        save_figure(image_rgb, mask_clean, results, out_png, show=show)
        save_csv(results, out_csv)
    return results, image_rgb


# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and classify wells across a timecourse for one bacteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir",
                        help="Folder containing segmented .png images.")
    parser.add_argument("output_dir",
                        help="Folder to write results into. Outputs go to "
                             "<output_dir>/<bacteria>/Three_Panel/ and "
                             "<output_dir>/<bacteria>/Composite/.")
    parser.add_argument("bacteria",
                        help="Bacteria name, e.g. bacteria. Files matching "
                             "<bacteria>_*.png in input_dir will be analyzed.")
    parser.add_argument("--max-width", type=int, default=1000,
                        help="Resize images so width does not exceed this (px). "
                             "Hough parameters above are tuned for ~1000 px wide "
                             "images; raise this only if your input is already small.")
    parser.add_argument("--timepoints", type=int, nargs="+", default=None,
                        metavar="T",
                        help="Only process specific timepoints (e.g. "
                             "--timepoints 0 5 10 15 20). If omitted, every "
                             "<bacteria>_*.png file in input_dir is processed.")
    parser.add_argument("--show", action="store_true",
                        help="Also display each plot interactively.")
    parser.add_argument("--composite", action="store_true",
                        help="Save a single composite figure of all timepoints "
                             "(skips per-timepoint figures and CSVs).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"input_dir does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    pattern = os.path.join(args.input_dir, f"{args.bacteria}_*.png")
    images  = sorted(glob.glob(pattern))
    if not images:
        print(f"No images matched: {pattern}", file=sys.stderr)
        sys.exit(1)

    # Optional --timepoints filter: keep only files whose <NN> is in the list.
    if args.timepoints is not None:
        wanted = set(args.timepoints)
        tp_re  = re.compile(rf"^{re.escape(args.bacteria)}_(\d+)$")
        def parse_tp(p):
            m = tp_re.match(os.path.splitext(os.path.basename(p))[0])
            return int(m.group(1)) if m else None
        filtered = [p for p in images if parse_tp(p) in wanted]
        found    = {parse_tp(p) for p in filtered}
        missing  = sorted(wanted - found)
        if missing:
            print(f"WARNING: requested timepoints not found in input: {missing}",
                  file=sys.stderr)
        if not filtered:
            print(f"No images at requested timepoints {sorted(wanted)}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Filtered {len(images)} -> {len(filtered)} images at "
              f"timepoints {sorted(found)} in {args.input_dir}")
        images = filtered
    else:
        print(f"Found {len(images)} images matching {args.bacteria}_*.png "
              f"in {args.input_dir}")

    composite_data = []

    for image_path in images:
        print(f"\n{'='*50}")
        results, image_rgb = run(image_path, args.output_dir, args.bacteria,
                                 max_width=args.max_width,
                                 show=args.show, composite_mode=args.composite)

        if args.composite:
            timepoint = os.path.splitext(os.path.basename(image_path))[0]
            composite_data.append((timepoint, image_rgb, results))

    if args.composite and composite_data:
        composite_out = os.path.join(args.output_dir, args.bacteria,
                                     "Composite", args.bacteria + "_composite.png")
        os.makedirs(os.path.dirname(composite_out), exist_ok=True)
        save_composite_figure(composite_data, composite_out, show=args.show)