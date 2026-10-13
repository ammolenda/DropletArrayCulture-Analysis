"""
Segment top/bottom device images into separate cropped images.

Reads .jpeg images from a single folder named:
    <date>_<bacteriatop>_<bacteriabottom>     e.g. day-month-year_bacteria1_bacteria2

Takes the first N timepoints (default 21) sorted by filename, crops out the
bright margins on the far left/right, splits each image horizontally at the
midpoint, and saves the two halves as PNGs into the output folder:

    <output_dir>/<bacteriatop>_<tt>.png      (top half)
    <output_dir>/<bacteriabottom>_<tt>.png   (bottom half)

Originals are never modified. Output is lossless PNG.

Usage:
    python segment_devices.py <input_dir> <output_dir> [-n 21] [--no-overwrite]

Examples:
    python segment_devices.py "/data/day-month-year_bacteria1_bacteria2" "/data/segmented"
    python segment_devices.py "/data/day-month-year_bacteria1_bacteria2" "/data/segmented" -n 21
"""

from pathlib import Path
import argparse
import sys
import numpy as np
from PIL import Image

# -------- TUNABLE DEFAULTS (rarely need to change) -----------------------
IMAGE_EXTS    = {".jpeg", ".jpg"}                # case-insensitive
SPLIT_FRAC    = 0.50                             # vertical split point (0.5 = midpoint)
AUTO_CROP_SIDES   = True                         # auto-detect chamber walls
BRIGHT_THRESHOLD  = 210                          # chamber interior is darker than this
SIDE_PAD_FRAC     = 0.015                        # extra margin added outward on each side
                                                 # after auto-detection, as a fraction of W
MANUAL_LEFT_FRAC  = 0.20                         # used only if AUTO_CROP_SIDES = False
MANUAL_RIGHT_FRAC = 0.21
TIME_FORMAT       = "{:02d}"                     # zero-padded: 00, 01, ..., 20
# -------------------------------------------------------------------------


def parse_dataset_folder(name: str):
    """Return (date, bacteria_top, bacteria_bottom) or None if name doesn't match."""
    parts = name.split("_")
    if len(parts) < 3:
        return None
    date = parts[0]
    bacteria_top = parts[1]
    bacteria_bottom = "_".join(parts[2:])  # tolerate underscores in the bottom name
    return date, bacteria_top, bacteria_bottom


def _find_longest_run(profile: np.ndarray) -> tuple[int | None, int | None]:
    """Bounds of the longest contiguous run of values below BRIGHT_THRESHOLD.
    Returns (start, end_exclusive) or (None, None) if no run exists. Picking
    the longest run (rather than the first) makes detection robust to
    whatever is outside the chamber: bright torn paper, dark background,
    handwriting, the ibidi logo, etc. — none of those produces a sustained
    dark band even close to the chamber's width."""
    below = profile < BRIGHT_THRESHOLD
    n = len(below)
    best_start = best_end = 0
    best_len = 0
    i = 0
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start, best_end = i, j
            i = j
        else:
            i += 1
    if best_len == 0:
        return None, None
    return best_start, best_end


def detect_side_bounds(img_array: np.ndarray) -> tuple[int, int]:
    """
    Find left/right x-bounds of the device region using two narrow horizontal
    bands (one through the top device, one through the bottom device) and
    taking the intersection. This gives a single x-range that keeps both
    halves clean of bright torn-paper margins. Falls back to manual fractions
    if detection fails.
    """
    gray = img_array if img_array.ndim == 2 else img_array.mean(axis=2)
    H, W = gray.shape

    top_band = gray[int(H * 0.18):int(H * 0.32), :].mean(axis=0)
    bot_band = gray[int(H * 0.65):int(H * 0.78), :].mean(axis=0)
    tL, tR = _find_longest_run(top_band)
    bL, bR = _find_longest_run(bot_band)

    if None in (tL, tR, bL, bR):
        return int(W * MANUAL_LEFT_FRAC), int(W * (1 - MANUAL_RIGHT_FRAC))

    left  = max(tL, bL)
    right = min(tR, bR)
    if right - left < W * 0.3:
        return int(W * MANUAL_LEFT_FRAC), int(W * (1 - MANUAL_RIGHT_FRAC))

    # add a small outward margin so the chamber wall isn't cropped flush
    pad = int(W * SIDE_PAD_FRAC)
    left  = max(0, left  - pad)
    right = min(W, right + pad)
    return left, right


def process_image(img_path: Path, out_dir: Path,
                  bacteria_top: str, bacteria_bottom: str, t: int,
                  overwrite: bool) -> None:
    """Crop sides, split horizontally, save top and bottom as PNGs."""
    with Image.open(img_path) as im:
        im.load()
        arr = np.asarray(im)

    H, W = arr.shape[:2]
    if AUTO_CROP_SIDES:
        x0, x1 = detect_side_bounds(arr)
    else:
        x0 = int(W * MANUAL_LEFT_FRAC)
        x1 = int(W * (1 - MANUAL_RIGHT_FRAC))

    y_split = int(H * SPLIT_FRAC)
    top    = arr[:y_split, x0:x1]
    bottom = arr[y_split:, x0:x1]

    tlabel = TIME_FORMAT.format(t)
    out_top    = out_dir / f"{bacteria_top}_{tlabel}.png"
    out_bottom = out_dir / f"{bacteria_bottom}_{tlabel}.png"

    if not overwrite and out_top.exists() and out_bottom.exists():
        print(f"  skip t={tlabel} (outputs exist)")
        return

    Image.fromarray(top).save(out_top, format="PNG", optimize=False, compress_level=1)
    Image.fromarray(bottom).save(out_bottom, format="PNG", optimize=False, compress_level=1)
    print(f"  t={tlabel}  {img_path.name}  ->  {out_top.name}, {out_bottom.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment top/bottom device images into separate cropped PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", type=Path,
                   help="Folder containing the .jpeg images. Folder name must "
                        "be in the form date_bacteriatop_bacteriabottom "
                        "(e.g. day-month-year_bacteria1_bacteria2).")
    p.add_argument("output_dir", type=Path,
                   help="Folder to write segmented PNGs into (created if missing).")
    p.add_argument("-n", "--n-timepoints", type=int, default=21,
                   help="Number of timepoints (images) to process.")
    p.add_argument("--no-overwrite", action="store_true",
                   help="Skip timepoints whose outputs already exist.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path  = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    overwrite = not args.no_overwrite

    if not input_dir.is_dir():
        print(f"input_dir does not exist: {input_dir}", file=sys.stderr)
        return 1

    parsed = parse_dataset_folder(input_dir.name)
    if parsed is None:
        print(f"Folder name '{input_dir.name}' does not match the expected "
              f"pattern date_bacteriatop_bacteriabottom (need at least two "
              f"underscores).", file=sys.stderr)
        return 1
    _, bacteria_top, bacteria_bottom = parsed

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No .jpeg images found in {input_dir}", file=sys.stderr)
        return 1
    images = images[:args.n_timepoints]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Pair:    top={bacteria_top}, bottom={bacteria_bottom}")
    print(f"Images:  {len(images)} (t = 0 .. {len(images) - 1})")

    for t, img_path in enumerate(images):
        try:
            process_image(img_path, output_dir, bacteria_top, bacteria_bottom,
                          t, overwrite)
        except Exception as e:
            print(f"  ERROR on {img_path.name} (t={t}): {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())