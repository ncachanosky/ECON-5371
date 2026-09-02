"""
Rebuild slide-only figure variants from already-rendered book PNGs.
Does NOT touch any chapter_XX.qmd or the book's rendered output — reads
existing PNGs under docs/chapters/... and writes new PNGs into
slides/chapters/figures/, for slide use only.

Two supported operations, driven by the JOBS list below:

  "split"   — crop a stacked figure into N separate output files
              (e.g. Dynamic Regimes: 7 panels -> 3 files; or a single
              panel pulled out of a 2-panel figure, one file)

  "compose" — crop a stacked figure into its individual panels, then
              recompose them into a grid (e.g. GDP ACF/PACF:
              4 stacked panels -> one 2x2 grid file)

Place this script at: slides/chapters/build_slide_figures.py
  (same folder as 01.qmd, 02.qmd, etc., so figures/ output sits right
  next to it)

Run any time the book's source figures change, or you need to
regenerate slide crops.

Usage (from repo root):
    python slides/chapters/build_slide_figures.py

    # Or run a single job by name:
    python slides/chapters/build_slide_figures.py --job diffeq_paths
    python slides/chapters/build_slide_figures.py --job arma_fanchart_only
"""

import argparse
from pathlib import Path
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent           # .../slides/chapters
REPO_ROOT  = SCRIPT_DIR.parents[1]                      # .../ECON-5371
OUT_DIR    = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BORDER_PX = 0   # spacing between panels in a composed grid (0 = flush, no border)


def book_figs(chapter: str) -> Path:
    """Path to a given chapter's rendered figure-html folder, e.g.
    book_figs("01") -> docs/chapters/01/chapter_01_files/figure-html
    book_figs("04") -> docs/chapters/04/chapter_04_files/figure-html"""
    return (REPO_ROOT / "docs" / "chapters" / chapter /
            f"chapter_{chapter}_files" / "figure-html")

# Kept for backward compatibility with existing Chapter 1 job definitions below.
BOOK_FIGS = book_figs("01")


# ── Shared crop utility ─────────────────────────────────────────────────
def crop_panels(img: Image.Image, n_panels: int,
                 top_margin_frac: float, pad_frac: float = 0.0) -> list[Image.Image]:
    """Split a vertically-stacked source image into n_panels equal-height crops."""
    w, h = img.size
    top_margin = int(h * top_margin_frac)
    usable_h = h - top_margin
    panel_h = usable_h / n_panels
    pad = int(pad_frac * h)

    panels = []
    for i in range(n_panels):
        top = max(0, int(top_margin + i * panel_h) - pad)
        bottom = min(h, int(top_margin + (i + 1) * panel_h) + pad)
        panels.append(img.crop((0, top, w, bottom)))
    return panels


def stack_vertical(panels: list[Image.Image]) -> Image.Image:
    """Stack panel images directly on top of each other, full width,
    no resizing or borders — used for the 'split' op where panels must
    stay pixel-faithful to the source crop (this is a straight vertical
    concatenation, not a grid layout)."""
    w = panels[0].width
    total_h = sum(p.height for p in panels)
    stacked = Image.new("RGB", (w, total_h), "white")
    y = 0
    for p in panels:
        stacked.paste(p, (0, y))
        y += p.height
    return stacked


def compose_grid(panels: list[Image.Image], n_cols: int) -> Image.Image:
    """Arrange a list of panel images into a grid with n_cols columns,
    filling row-major (left-to-right, top-to-bottom). Used for the
    'compose' op, where panels are resized to a common width to align
    into columns — appropriate there since source panels are already
    similar dimensions; NOT used for 'split', which needs pixel-faithful
    stacking instead (see stack_vertical)."""
    n_rows = -(-len(panels) // n_cols)  # ceil division

    min_w = min(p.width for p in panels)
    panels = [p.resize((min_w, int(p.height * min_w / p.width))) for p in panels]

    row_heights = []
    for r in range(n_rows):
        row_panels = panels[r * n_cols:(r + 1) * n_cols]
        row_heights.append(max(p.height for p in row_panels))

    grid_w = min_w * n_cols + BORDER_PX * (n_cols + 1)
    grid_h = sum(row_heights) + BORDER_PX * (n_rows + 1)
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    y_offset = BORDER_PX
    for r in range(n_rows):
        x_offset = BORDER_PX
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(panels):
                break
            grid.paste(panels[idx], (x_offset, y_offset))
            x_offset += min_w + BORDER_PX
        y_offset += row_heights[r] + BORDER_PX

    return grid


# ── Job definitions ──────────────────────────────────────────────────────
# Add a new figure here rather than writing a new script.
JOBS = {

    "diffeq_paths": {
        "op": "split",
        "source": BOOK_FIGS / "fig-diffeq-paths-output-1.png",
        "n_panels": 7,
        "top_margin_frac": 0.035,
        "pad_frac": 0.01,
        # (start_panel_idx, end_panel_idx, output_filename) — 0-indexed,
        # inclusive. (a)=0 ... (g)=6
        "groups": [
            (0, 2, "fig-diffeq-paths_stable.png"),
            (3, 4, "fig-diffeq-paths_unitroot.png"),
            (5, 6, "fig-diffeq-paths_explosive.png"),
        ],
    },

    "acfpacf_gdp": {
        "op": "compose",
        "source": BOOK_FIGS / "fig-acfpacf-gdp-output-1.png",
        "n_panels": 4,
        "top_margin_frac": 0.025,
        "pad_frac": 0.0,
        "n_cols": 2,   # panels fill row-major: [0,1] top row, [2,3] bottom row
        "output": "fig-acfpacf-gdp_grid.png",
    },

    "transformations": {
        "op": "compose",
        "source": BOOK_FIGS / "fig-transformations-output-1.png",
        "n_panels": 4,
        "top_margin_frac": 0.025,
        "pad_frac": 0.0,
        "n_cols": 2,   # [0,1] top row (GDP level, GDP growth),
                       # [2,3] bottom row (CPI level, CPI inflation)
        "output": "fig-transformations_grid.png",
    },

    "hp_hamilton": {
        "op": "compose",
        "source": book_figs("02") / "fig-hp-hamilton-output-1.png",
        "n_panels": 4,
        "top_margin_frac": 0.02,
        "pad_frac": 0.0,
        "n_cols": 2,   # [0,1] top row (HP filter, Hamilton filter),
                       # [2,3] bottom row (linear detrend, STL irregular)
        "output": "fig-hp-hamilton_grid.png",
    },

    # ── Side-by-side forecast comparison (Chapter 4, Section 4 closing slide) ──
    # Each source figure is a 2-panel stack: [0] point-forecast-only panel,
    # [1] fan chart panel. The comparison slide only needs the fan chart —
    # showing both full figures side by side was too crowded. "split" with
    # a single-entry group keeps just panel index 1.
    "arma_fanchart_only": {
        "op": "split",
        "source": book_figs("03") / "fig-forecast-output-1.png",
        "n_panels": 2,
        "top_margin_frac": 0.04,
        "pad_frac": 0.01,
        "groups": [
            (1, 1, "fig-forecast_fanchart-only.png"),  # keep only panel 1 (bottom = fan chart)
        ],
    },

    "arima_fanchart_only": {
        "op": "split",
        "source": book_figs("04") / "fig-arima-forecast-output-1.png",
        "n_panels": 2,
        "top_margin_frac": 0.04,
        "pad_frac": 0.01,
        "groups": [
            (1, 1, "fig-arima-forecast_fanchart-only.png"),
        ],
    },

}


def run_job(name: str, cfg: dict):
    source = cfg["source"]
    if not source.exists():
        print(f"[{name}] SKIPPED — source not found: {source}")
        print(f"          Render the book first, then re-run.")
        return

    img = Image.open(source)
    print(f"[{name}] source: {img.size[0]}x{img.size[1]}px")

    panels = crop_panels(img, cfg["n_panels"], cfg["top_margin_frac"],
                          cfg.get("pad_frac", 0.0))

    if cfg["op"] == "split":
        for start_idx, end_idx, out_name in cfg["groups"]:
            group_panels = panels[start_idx:end_idx + 1]
            stacked = stack_vertical(group_panels)
            out_path = OUT_DIR / out_name
            stacked.save(out_path)
            print(f"  saved {out_path.name}  panels {start_idx}-{end_idx}  {stacked.size}")

    elif cfg["op"] == "compose":
        grid = compose_grid(panels, n_cols=cfg["n_cols"])
        out_path = OUT_DIR / cfg["output"]
        grid.save(out_path)
        print(f"  saved {out_path.name}  {grid.size}")

    else:
        raise ValueError(f"Unknown op '{cfg['op']}' for job '{name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=list(JOBS.keys()), default=None,
                         help="Run only this job (default: run all jobs)")
    args = parser.parse_args()

    jobs_to_run = {args.job: JOBS[args.job]} if args.job else JOBS

    for name, cfg in jobs_to_run.items():
        run_job(name, cfg)

    print("\nDone. Check each output for panel-title/legend clipping — "
          "adjust the relevant job's top_margin_frac or pad_frac and re-run "
          "if anything looks cut off.")


if __name__ == "__main__":
    main()