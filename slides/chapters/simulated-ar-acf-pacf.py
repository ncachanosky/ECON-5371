"""
Simulate AR(1), AR(2), AR(3), MA(1), and MA(2) processes with made-up
(pedagogically chosen) coefficients, and plot each one's ACF/PACF side by
side — one figure per process, for one slide per process.

This is illustrative data — NOT drawn from the book's real series. It
exists purely to give students a direct visual link between a model's
order/family and its ACF/PACF signature: AR cuts off in the PACF at lag p;
MA cuts off in the ACF at lag q.

Does NOT touch chapter_03.qmd or any book content — this is a slides-only
figure, generated fresh rather than cropped from book output.

Place this script at: slides/chapters/simulate_ar_acf_pacf.py
  (same folder as 03.qmd)

Usage (from repo root):
    python slides/chapters/simulate_ar_acf_pacf.py
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # .../slides/chapters
OUT_DIR    = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── EO palette (matches the book's Python setup conventions) ──────────────
EO_CHARCOAL   = "#36454F"
EO_COPPER     = "#B87333"
EO_SAGE       = "#87A96B"
EO_SKYBLUE    = "#5B9BD5"
EO_TERRACOTTA = "#D4745E"
EO_LAVENDER   = "#8E7AB5"
PAGE_BG       = "#FAFAF8"

mpl.rcParams.update({
    "figure.facecolor":  PAGE_BG,
    "figure.edgecolor":  PAGE_BG,
    "axes.facecolor":    PAGE_BG,
    "axes.edgecolor":    EO_CHARCOAL,
    "axes.linewidth":    0.7,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   EO_CHARCOAL,
    "axes.titlelocation": "left",
    "axes.labelsize":    9,
    "axes.labelcolor":   EO_CHARCOAL,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.5,
    "grid.linestyle":    "--",
    "grid.alpha":        0.8,
    "xtick.color":       EO_CHARCOAL,
    "ytick.color":       EO_CHARCOAL,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.3,
    "font.family":       "serif",
    "font.serif":        ["Palatino Linotype", "Palatino", "Georgia", "DejaVu Serif"],
    "font.sans-serif":   ["Calibri", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "text.color":        EO_CHARCOAL,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": PAGE_BG,
})


def eo_style_ax(ax):
    for obj in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        obj.set_fontfamily("Calibri")


def eo_suptitle(fig, title, **kwargs):
    defaults = dict(fontsize=12, fontweight="bold",
                     color=EO_CHARCOAL, fontfamily="Calibri", y=1.03)
    defaults.update(kwargs)
    fig.suptitle(title, **defaults)


# ── Pedagogically chosen coefficients ──────────────────────────────────────
# Kept deliberately simple and stable/invertible (all roots well inside the
# unit circle) so the ACF/PACF signatures are unambiguous on a projector.
#
# statsmodels ArmaProcess uses the convention:
#   ar = [1, -phi_1, -phi_2, ...]   for the AR polynomial
#   ma = [1,  theta_1, theta_2, ...] for the MA polynomial
SPECS = [
    {"kind": "AR", "order": 1, "coefs": [0.7],            "color": EO_COPPER,
     "label": "AR(1): $\\phi_1=0.7$", "out_name": "fig-ar1-acf-pacf-simulated.png"},
    {"kind": "AR", "order": 2, "coefs": [0.6, -0.3],       "color": EO_SAGE,
     "label": "AR(2): $\\phi_1=0.6,\\ \\phi_2=-0.3$", "out_name": "fig-ar2-acf-pacf-simulated.png"},
    {"kind": "AR", "order": 3, "coefs": [0.5, 0.2, -0.25], "color": EO_SKYBLUE,
     "label": "AR(3): $\\phi_1=0.5,\\ \\phi_2=0.2,\\ \\phi_3=-0.25$", "out_name": "fig-ar3-acf-pacf-simulated.png"},

    {"kind": "MA", "order": 1, "coefs": [0.7],            "color": EO_TERRACOTTA,
     "label": "MA(1): $\\theta_1=0.7$", "out_name": "fig-ma1-acf-pacf-simulated.png"},
    {"kind": "MA", "order": 2, "coefs": [0.6, 0.3],        "color": EO_LAVENDER,
     "label": "MA(2): $\\theta_1=0.6,\\ \\theta_2=0.3$", "out_name": "fig-ma2-acf-pacf-simulated.png"},
]

N_OBS   = 500     # simulated series length (long enough for stable sample ACF/PACF)
N_LAGS  = 20
SEED    = 42


def make_figure(spec, rng):
    if spec["kind"] == "AR":
        ar_coefs = np.r_[1, -np.array(spec["coefs"])]   # AR polynomial
        ma_coefs = np.r_[1]                              # no MA part
    elif spec["kind"] == "MA":
        ar_coefs = np.r_[1]                              # no AR part
        ma_coefs = np.r_[1, np.array(spec["coefs"])]      # MA polynomial
    else:
        raise ValueError(f"Unknown kind '{spec['kind']}'")

    process = ArmaProcess(ar_coefs, ma_coefs)

    assert process.isstationary, f"{spec['kind']}({spec['order']}) spec is not stationary!"
    if spec["kind"] == "MA":
        assert process.isinvertible, f"MA({spec['order']}) spec is not invertible!"

    y = process.generate_sample(nsample=N_OBS, distrvs=rng.standard_normal)

    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(9, 3.6))

    plot_acf(y, lags=N_LAGS, ax=ax_acf, color=spec["color"],
              vlines_kwargs={"colors": spec["color"]},
              title="ACF", zero=False, alpha=0.05)

    plot_pacf(y, lags=N_LAGS, ax=ax_pacf, color=spec["color"],
               vlines_kwargs={"colors": spec["color"]}, method="ywm",
               title="PACF", zero=False, alpha=0.05)

    for ax in (ax_acf, ax_pacf):
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1.05, 1.05)
        eo_style_ax(ax)
        for line in ax.lines:
            if line.get_linestyle() == "--":
                line.set_color(EO_TERRACOTTA)
                line.set_linewidth(0.7)

    eo_suptitle(fig, spec["label"])
    fig.tight_layout()
    return fig


def main():
    rng = np.random.default_rng(SEED)

    for spec in SPECS:
        fig = make_figure(spec, rng)
        out_path = OUT_DIR / spec["out_name"]
        fig.savefig(out_path)
        print(f"Saved {out_path}  ({fig.get_size_inches()}in @ {mpl.rcParams['savefig.dpi']}dpi)")
        plt.close(fig)


if __name__ == "__main__":
    main()