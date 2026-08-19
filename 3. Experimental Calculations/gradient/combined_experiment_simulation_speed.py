#!/usr/bin/env python3
"""
Combined experiment + simulation speed plot for single and double PMAS.

Curves:
    1. single PMAS experiment  -> boxplots + smooth mean curve
    2. double PMAS experiment  -> boxplots + smooth mean curve
    3. single PMAS simulation  -> dashed line
    4. double PMAS simulation  -> dashed line

Inputs:
    speed_vs_dist_csv_sing/*.csv
    speed_vs_dist_csv_doub/*.csv
    static_gradient_speed_single_vs_double.csv

Outputs:
    averaged_plots/experiment_vs_static_simulation_speed.pdf
    averaged_plots/experiment_vs_static_simulation_speed.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import UnivariateSpline


# ============================================================
# USER SETTINGS
# ============================================================

SINGLE_FOLDER = Path("speed_vs_dist_csv_sing")
DOUBLE_FOLDER = Path("speed_vs_dist_csv_doub")
SIM_CSV = Path("sim_static_gradient_speed_single_vs_double.csv")

DIST_COL = "distance_cm"
SPEED_COL = "speed_mm_s"

SIM_DIST_COL = "distance_to_center_cm"
SIM_SPEED_COL = "speed_mm_s"
SIM_CASE_COL = "case"

BIN_SIZE_CM = 0.4
MIN_POINTS_PER_BIN = 5

SPLINE_SMOOTHING_SINGLE = 0.1
SPLINE_SMOOTHING_DOUBLE = 0.05

DIST_MAX = 10.0

OUTPUT_DIR = Path("averaged_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_NAME = "experiment_vs_static_simulation_speed"

# Label positions; tune these after first plot
LABEL_POSITIONS = {
    "single_exp":  {"x": 1.5, "y": 0.15},
    "double_exp":  {"x": 2.6, "y": 1.4},
    "single_sim":  {"x": 2.6, "y": 2.4},
    "double_sim":  {"x": 3.2, "y": 3.5},
}

# Use the same colors as before
COLOR_SINGLE = "black"
COLOR_DOUBLE = "red"


# ============================================================
# DATA LOADING
# ============================================================

def load_csv_folder(folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)

        if DIST_COL not in df.columns or SPEED_COL not in df.columns:
            raise ValueError(
                f"\nColumn problem in {file}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Expected columns: {DIST_COL}, {SPEED_COL}"
            )

        temp = df[[DIST_COL, SPEED_COL]].copy()
        temp["source_file"] = file.name
        all_data.append(temp)

    data = pd.concat(all_data, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[DIST_COL, SPEED_COL])

    return data


def load_simulation_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find simulation CSV: {path}")

    df = pd.read_csv(path)

    required = {SIM_CASE_COL, SIM_DIST_COL, SIM_SPEED_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"\nSimulation CSV is missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[[SIM_CASE_COL, SIM_DIST_COL, SIM_SPEED_COL]].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[SIM_CASE_COL, SIM_DIST_COL, SIM_SPEED_COL])

    df = df[df[SIM_DIST_COL] <= DIST_MAX]

    return df


# ============================================================
# BINNING
# ============================================================

def bin_by_distance(data: pd.DataFrame, bin_size_cm: float):
    d_min = np.floor(data[DIST_COL].min() / bin_size_cm) * bin_size_cm
    d_max = np.ceil(data[DIST_COL].max() / bin_size_cm) * bin_size_cm

    bins = np.arange(d_min, d_max + bin_size_cm, bin_size_cm)

    data = data.copy()
    data["distance_bin"] = pd.cut(
        data[DIST_COL],
        bins=bins,
        include_lowest=True,
        right=False,
    )

    grouped = data.groupby("distance_bin", observed=True)

    bin_centers = []
    bin_values = []

    for _, group in grouped:
        if len(group) < MIN_POINTS_PER_BIN:
            continue

        bin_center = group[DIST_COL].mean()
        values = group[SPEED_COL].to_numpy()

        bin_centers.append(bin_center)
        bin_values.append(values)

    return np.array(bin_centers), bin_values


# ============================================================
# FITTING
# ============================================================

def spline_fit(x, y, smoothing):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]

    if len(unique_x) < 4:
        return None, None

    spline = UnivariateSpline(unique_x, unique_y, s=smoothing)

    x_fit = np.linspace(unique_x.min(), unique_x.max(), 500)
    y_fit = spline(x_fit)

    return x_fit, y_fit


# ============================================================
# PLOTTING HELPERS
# ============================================================

def add_text_label(ax, text, color, key):
    ax.text(
        LABEL_POSITIONS[key]["x"],
        LABEL_POSITIONS[key]["y"],
        text,
        color=color,
        fontsize=10,
        ha="left",
        va="center",
    )


def plot_experiment_dataset(
    ax,
    bin_centers,
    bin_values,
    label,
    color,
    smoothing,
    label_key,
):
    ax.boxplot(
        bin_values,
        positions=bin_centers,
        widths=BIN_SIZE_CM * 0.7,
        patch_artist=False,
        showfliers=False,
        manage_ticks=False,
        boxprops=dict(color=color, linewidth=1),
        whiskerprops=dict(color=color, linewidth=1),
        capprops=dict(color=color, linewidth=1),
        medianprops=dict(color=color, linewidth=1.5),
    )

    y_mean = np.array([np.mean(v) for v in bin_values])

    x_fit, y_fit = spline_fit(
        bin_centers,
        y_mean,
        smoothing,
    )

    if x_fit is not None:
        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linewidth=1.2,
            linestyle="-",
        )

    add_text_label(ax, label, color, label_key)


def plot_simulation_dataset(
    ax,
    sim_df,
    case_name,
    label,
    color,
    label_key,
):
    data = sim_df[sim_df[SIM_CASE_COL] == case_name].copy()

    if data.empty:
        raise ValueError(
            f"No simulation rows found for case '{case_name}'. "
            f"Available cases: {sim_df[SIM_CASE_COL].unique()}"
        )

    data = data.sort_values(SIM_DIST_COL)

    ax.plot(
        data[SIM_DIST_COL],
        data[SIM_SPEED_COL],
        color=color,
        linewidth=1.6,
        linestyle="--",
    )

    add_text_label(ax, label, color, label_key)


# ============================================================
# MAIN PLOT
# ============================================================

def make_combined_plot():
    single_data = load_csv_folder(SINGLE_FOLDER)
    double_data = load_csv_folder(DOUBLE_FOLDER)
    sim_data = load_simulation_csv(SIM_CSV)

    single_data = single_data[single_data[DIST_COL] <= DIST_MAX]
    double_data = double_data[double_data[DIST_COL] <= DIST_MAX]

    single_centers, single_values = bin_by_distance(single_data, BIN_SIZE_CM)
    double_centers, double_values = bin_by_distance(double_data, BIN_SIZE_CM)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    # --------------------------------------------------------
    # Experimental boxplots + smooth curves
    # --------------------------------------------------------
    plot_experiment_dataset(
        ax=ax,
        bin_centers=single_centers,
        bin_values=single_values,
        label="single PMAS experiment",
        color=COLOR_SINGLE,
        smoothing=SPLINE_SMOOTHING_SINGLE,
        label_key="single_exp",
    )

    plot_experiment_dataset(
        ax=ax,
        bin_centers=double_centers,
        bin_values=double_values,
        label="double PMAS experiment",
        color=COLOR_DOUBLE,
        smoothing=SPLINE_SMOOTHING_DOUBLE,
        label_key="double_exp",
    )

    # --------------------------------------------------------
    # Static-gradient simulation curves
    # --------------------------------------------------------
    plot_simulation_dataset(
        ax=ax,
        sim_df=sim_data,
        case_name="single PMAS",
        label="single PMAS simulation",
        color=COLOR_SINGLE,
        label_key="single_sim",
    )

    plot_simulation_dataset(
        ax=ax,
        sim_df=sim_data,
        case_name="double PMAS",
        label="double PMAS simulation",
        color=COLOR_DOUBLE,
        label_key="double_sim",
    )

    # --------------------------------------------------------
    # Axes and style
    # --------------------------------------------------------
    ax.set_xlabel("Distance / cm")
    ax.set_ylabel("Speed / mm/s")

    ax.set_xlim(left=0, right=DIST_MAX)

    all_speed = pd.concat(
        [
            single_data[SPEED_COL],
            double_data[SPEED_COL],
            sim_data[SIM_SPEED_COL],
        ],
        ignore_index=True,
    )

    ax.set_ylim(bottom=0, top=4.5)

    ax.tick_params(which="both", top=True, right=True)

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / f"{OUTPUT_NAME}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_NAME}.png"

    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, dpi=300)

    plt.show()

    print("Saved:")
    print(f"  - {pdf_path}")
    print(f"  - {png_path}")


if __name__ == "__main__":
    make_combined_plot()