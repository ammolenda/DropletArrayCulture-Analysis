"""
Collate Well Detection Results
==============================
Walks a folder of well-detection CSVs (output of well_detector.py), pools
results across all bacteria/replicates/timepoints, and produces:

  1. A wide-format CSV of yellow fraction vs. time (one column per bacteria,
     one row per well per replicate), formatted for import into GraphPad
     Prism as an XY scatter table.
  2. A table of positive-droplet counts per (bacteria, replicate) per
     timepoint, written as both CSV and PNG, and printed to the terminal.

The script accepts either:
  * A folder hierarchy like
        <input>/<experiment>/<bacteria>/Three_Panel/<bacteria>_<tt>_results.csv
    in which case the experiment name (the folder above the bacteria folder)
    is used to distinguish replicates, OR
  * A flat folder containing CSVs named <bacteria>_<tt>_results.csv, in
    which case all CSVs of a given bacteria are treated as one replicate.

Usage:
    python collate_results.py <input_dir> <output_dir> [--threshold 0.25]

Examples:
    python collate_results.py /data/all_analyzed /data/summary
    python collate_results.py /data/all_analyzed /data/summary --threshold 0.30
"""

from pathlib import Path
import argparse
import csv
import re
import sys

import matplotlib.pyplot as plt


# -------- defaults ------------------------------------------------------
DEFAULT_THRESHOLD = 0.25     # match well_detector.POSITIVE_THRESHOLD
DPI               = 150
# ------------------------------------------------------------------------

# Canonical display names for the bacteria abbreviations used in folder/file
# names (e.g. "sepi" -> "S. epidermidis"). Only used to make CSV headers
# readable. Keys are matched case-insensitively.
NAME_ALIASES = {
    "abaumannii":     "A. baumannii",
    "a_baumannii":    "A. baumannii",
    "ab":             "A. baumannii",
    "ecoli":          "E. coli",
    "e_coli":         "E. coli",
    "ec":             "E. coli",
    "kpneumoniae":    "K. pneumoniae",
    "k_pneumoniae":   "K. pneumoniae",
    "kp":             "K. pneumoniae",
    "paeruginosa":    "P. aeruginosa",
    "p_aeruginosa":   "P. aeruginosa",
    "pa":             "P. aeruginosa",
    "efaecalis":      "E. faecalis",
    "e_faecalis":     "E. faecalis",
    "ef":             "E. faecalis",
    "mrsa":           "MRSA",
    "sepidermidis":   "S. epidermidis",
    "s_epidermidis":  "S. epidermidis",
    "sepi":           "S. epidermidis",
}


def resolve_display_name(raw_name: str) -> str:
    """Return the canonical display name for a bacteria abbreviation
    (e.g. 'sepi' -> 'S. epidermidis'), or the raw name unchanged if it
    isn't a recognized abbreviation."""
    return NAME_ALIASES.get(raw_name.lower(), raw_name)
# ------------------------------------------------------------------------


def parse_csv_filename(name: str):
    """Return (bacteria, timepoint) from '<bacteria>_<tt>_results.csv'."""
    stem = name[:-len(".csv")] if name.endswith(".csv") else name
    if stem.endswith("_results"):
        stem = stem[:-len("_results")]
    m = re.match(r"^(.+)_(\d+)$", stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def get_experiment_id(csv_path: Path, bacteria: str) -> str:
    """Walk up from the CSV looking for a folder named '<bacteria>'; the
    folder above that is the experiment. Return '-' if the layout is flat."""
    for parent in csv_path.parents:
        if parent.name == bacteria:
            grand = parent.parent
            return grand.name if grand and grand.name else "-"
    return "-"


def load_records(input_dir: Path, threshold: float):
    """Recursively scan input_dir for *_results.csv files. Return a list of
    per-well records and a sorted list of timepoints actually seen."""
    records = []
    timepoints_seen = set()
    csv_paths = sorted(input_dir.rglob("*_results.csv"))
    for csv_path in csv_paths:
        parsed = parse_csv_filename(csv_path.name)
        if parsed is None:
            print(f"  skip (filename doesn't match pattern): {csv_path.name}")
            continue
        bacteria, timepoint = parsed
        experiment = get_experiment_id(csv_path, bacteria)
        timepoints_seen.add(timepoint)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fraction = float(row["yellow_fraction"])
                except (KeyError, ValueError):
                    continue
                records.append({
                    "bacteria":   bacteria,
                    "experiment": experiment,
                    "timepoint":  timepoint,
                    "well":       int(row.get("well_number", 0)),
                    "fraction":   fraction,
                    "positive":   fraction >= threshold,
                })
    return records, sorted(timepoints_seen)


def save_scatter_data_csv(records, out_path: Path) -> None:
    """Wide-format CSV for GraphPad Prism XY-scatter import.

    Layout:
        Time, <Bacteria 1>, <Bacteria 2>, ...
        0,    0.95,         0.85
        0,    0.05,         0.10
        ...                                  (one row per well per replicate)

    Each row carries one well's yellow-fraction reading at that timepoint.
    Multiple rows share the same Time value -- Prism interprets this as
    replicates at that X. Columns of unequal length (different replicate
    counts per bacteria) are padded with blanks. Headers use canonical
    display names (e.g. 'S. epidermidis') for readability.
    """
    bacteria_list = sorted({r["bacteria"] for r in records})
    timepoints    = sorted({r["timepoint"] for r in records})

    # group fractions by (bacteria, timepoint)
    by_bt = {}
    for r in records:
        by_bt.setdefault((r["bacteria"], r["timepoint"]), []).append(r["fraction"])

    # display names (canonical where available)
    display_names = [resolve_display_name(raw) for raw in bacteria_list]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time"] + display_names)

        for t in timepoints:
            max_wells = max((len(by_bt.get((b, t), [])) for b in bacteria_list),
                            default=0)
            for i in range(max_wells):
                row = [t]
                for b in bacteria_list:
                    fracs = by_bt.get((b, t), [])
                    row.append(f"{fracs[i]:.4f}" if i < len(fracs) else "")
                w.writerow(row)
    print(f"  scatter data CSV saved: {out_path}")


def compute_count_table(records, timepoints):
    """Return (replicates_sorted, table) where table[(b, exp)][t] = n_positive."""
    replicates = sorted({(r["bacteria"], r["experiment"]) for r in records})
    table = {rep: {t: 0 for t in timepoints} for rep in replicates}
    for r in records:
        if r["positive"]:
            table[(r["bacteria"], r["experiment"])][r["timepoint"]] += 1
    return replicates, table


def save_count_table_csv(replicates, timepoints, table, out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bacteria", "experiment"] + [f"t{t:02d}" for t in timepoints])
        for rep in replicates:
            bacteria, experiment = rep
            w.writerow([bacteria, experiment] + [table[rep][t] for t in timepoints])
    print(f"  counts CSV saved: {out_path}")


def save_count_table_png(replicates, timepoints, table, out_path: Path) -> None:
    n_rows = len(replicates)
    n_cols = len(timepoints) + 2  # bacteria + experiment + per-timepoint

    # estimate text widths (in characters) for sizing
    bact_chars = max(len("Bacteria"),   max(len(r[0]) for r in replicates))
    exp_chars  = max(len("Experiment"), max(len(r[1]) for r in replicates))
    num_chars  = max(3, max(len(f"t{t:02d}") for t in timepoints))

    # convert char counts to relative column widths (cells_widths must sum to <= 1.0)
    raw_widths = ([bact_chars + 2, exp_chars + 2]
                  + [num_chars + 1] * len(timepoints))
    total = sum(raw_widths)
    col_widths = [w / total for w in raw_widths]

    fig_w = max(10.0, 0.13 * total + 1.5)
    fig_h = max(2.0, 0.5 * n_rows + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_headers = ["Bacteria", "Experiment"] + [f"t{t:02d}" for t in timepoints]
    cell_text = []
    for rep in replicates:
        bacteria, experiment = rep
        cell_text.append([bacteria, experiment]
                         + [str(table[rep][t]) for t in timepoints])

    tbl = ax.table(cellText=cell_text, colLabels=col_headers,
                   colWidths=col_widths,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    # left-align the bacteria and experiment cells; bold the header
    for c in range(n_cols):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#e8e8e8")
    for r in range(1, n_rows + 1):
        tbl[(r, 0)].set_text_props(ha="left")
        tbl[(r, 0)].PAD = 0.04
        tbl[(r, 1)].set_text_props(ha="left")
        tbl[(r, 1)].PAD = 0.04

    ax.set_title("Positive droplets per timepoint", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  counts PNG saved: {out_path}")


def print_count_table(replicates, timepoints, table) -> None:
    if not replicates:
        return
    bw = max(len("Bacteria"),    max(len(r[0]) for r in replicates))
    ew = max(len("Experiment"),  max(len(r[1]) for r in replicates))
    tw = 4  # column width for each timepoint
    head = f"{'Bacteria':<{bw}}  {'Experiment':<{ew}}  " + " ".join(
        f"{('t'+str(t).zfill(2)):>{tw}}" for t in timepoints)
    print()
    print(head)
    print("-" * len(head))
    for rep in replicates:
        bacteria, experiment = rep
        cells = " ".join(f"{table[rep][t]:>{tw}d}" for t in timepoints)
        print(f"{bacteria:<{bw}}  {experiment:<{ew}}  {cells}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collate well_detector CSVs into a scatter plot and "
                    "a positive-count table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", type=Path,
                   help="Folder of CSVs (any depth; *_results.csv are found "
                        "recursively).")
    p.add_argument("output_dir", type=Path,
                   help="Folder to write summary outputs into "
                        "(scatter plot + count table CSV/PNG).")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Yellow-fraction threshold for calling a well positive. "
                        "Wells with fraction >= threshold are counted as positive.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir  = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"input_dir does not exist: {input_dir}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"Threshold: {args.threshold}")

    print("Loading CSVs...")
    records, timepoints = load_records(input_dir, args.threshold)
    if not records:
        print("No usable records found.", file=sys.stderr)
        return 1
    bacteria = sorted({r["bacteria"] for r in records})
    print(f"  {len(records)} well rows, "
          f"{len(bacteria)} bacteria ({', '.join(bacteria)}), "
          f"{len(timepoints)} timepoints "
          f"(t{min(timepoints):02d} .. t{max(timepoints):02d})")

    print("Saving scatter data CSV...")
    save_scatter_data_csv(records,
                          output_dir / "scatter_yellow_fraction.csv")

    print("Computing positive-count table...")
    replicates, table = compute_count_table(records, timepoints)
    save_count_table_csv(replicates, timepoints, table,
                         output_dir / "positive_counts.csv")
    save_count_table_png(replicates, timepoints, table,
                         output_dir / "positive_counts.png")
    print_count_table(replicates, timepoints, table)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())