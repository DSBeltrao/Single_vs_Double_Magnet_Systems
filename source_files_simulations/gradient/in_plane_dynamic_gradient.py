import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import magpylib as magpy
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator

# ============================================================
# SETTINGS
# ============================================================
# Input CSVs. The double-magnet CSV keeps its original name.
CSV_SINGLE = "commands_open_loop_single_magnet.csv"
CSV_DOUBLE = "commands_open_loop.csv"

OUT_PNG = "gradient_direction_comparison_2x3.png"
OUT_SINGLE_CSV = "gradient_direction_quantitative_single_magnet.csv"
OUT_DOUBLE_CSV = "gradient_direction_quantitative_double_magnet.csv"

# Easy global plot tuning
FONT_SIZE = 25              # axis labels, tick numbers, row/column titles
LABEL_FONT_SIZE = 25        # direct labels on the plotted curves
TITLE_FONT_SIZE = 25
FIGSIZE = (24.0, 8.0)       # 3 columns x 2 rows
LW = 2.0
DPI = 300

# Geometry / magnets
POOL_SIZE_CM = 15.0
MAGNET_SIDE_CM = 4.0
MAGNET_OFFSET_Z_CM = 12.5
BR_T = 1.3
H = 1e-4                    # finite-difference step [m]

# Robot case
ROBOT_CASE = "THESIS_10MM"  # "THESIS_10MM" or "SCALED_100UM"

# Label smoothing and placement.
# dx/dy are in points, so they are easy to tune without changing data coordinates.
LABEL_SMOOTH_WINDOW = 15
LABELS = {
    "single": {
        "gradient speed magnitude": {"frac": 0.3, "dx": 8, "dy": 30, "rotation": 0, "ha": "left", "va": "center"},
        "absolute direction error": {"frac": 0.6, "dx": 8, "dy": -80, "rotation": 0, "ha": "left", "va": "center"},
        "along-path": {"frac": 0.68, "dx": -10, "dy": -30, "rotation": 0, "ha": "left", "va": "center"},
        "lateral": {"frac": 0.68, "dx": -10, "dy": 30, "rotation": 0, "ha": "left", "va": "center"},
    },
    "double": {
        "gradient speed magnitude": {"frac": 0.6, "dx": 8, "dy": -80, "rotation": 0, "ha": "left", "va": "center"},
        "absolute direction error": {"frac": 0.6, "dx": 8, "dy": -45, "rotation": 0, "ha": "left", "va": "center"},
        "along-path": {"frac": 0.05, "dx": -10, "dy": -120, "rotation": 0, "ha": "left", "va": "center"},
        "lateral": {"frac": 0.8, "dx": -8, "dy": 85, "rotation": 0, "ha": "left", "va": "center"},
    },
}

# ============================================================
# ROBOT MOMENT / DRAG
# ============================================================
M_NDFEB_A_PER_M = 1.03e6
V_MAG_M3 = 2.36e-9
MR_DIPOLE_MOMENT_10MM_AM2 = M_NDFEB_A_PER_M * V_MAG_M3
SCALE_LAMBDA = 0.01
MR_DIPOLE_MOMENT_100UM_AM2 = MR_DIPOLE_MOMENT_10MM_AM2 * SCALE_LAMBDA**3


def perrin_translational_drag_ns_per_m(eta_pa_s, a_m, b_m):
    if a_m <= 0 or b_m <= 0:
        raise ValueError("Semi-axes must be positive.")
    if a_m < b_m:
        raise ValueError("Use prolate spheroid with a >= b.")

    p = a_m / b_m
    if np.isclose(p, 1.0):
        return 6.0 * np.pi * eta_pa_s * a_m

    xi = np.sqrt(p**2 - 1.0) / p
    S = 2.0 * np.arctanh(xi) / xi
    V = (4.0 / 3.0) * np.pi * a_m * b_m**2
    R_eff = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    f_sphere = 6.0 * np.pi * eta_pa_s * R_eff
    f_perrin = 2.0 * p**(2.0 / 3.0) / S
    return f_sphere * f_perrin


ETA_GLYCERIN_PA_S = 1.14
A_ELLIP_10MM_M = 5.0e-3
B_ELLIP_10MM_M = 3.25e-3
TRANSLATIONAL_DRAG_10MM_NS_PER_M = perrin_translational_drag_ns_per_m(
    ETA_GLYCERIN_PA_S, A_ELLIP_10MM_M, B_ELLIP_10MM_M
)

ETA_BLOOD_PA_S = 5e-3
A_ELLIP_100UM_M = A_ELLIP_10MM_M * SCALE_LAMBDA
B_ELLIP_100UM_M = B_ELLIP_10MM_M * SCALE_LAMBDA
TRANSLATIONAL_DRAG_100UM_NS_PER_M = perrin_translational_drag_ns_per_m(
    ETA_BLOOD_PA_S, A_ELLIP_100UM_M, B_ELLIP_100UM_M
)

if ROBOT_CASE == "THESIS_10MM":
    MR_DIPOLE_MOMENT_AM2 = MR_DIPOLE_MOMENT_10MM_AM2
    TRANSLATIONAL_DRAG_NS_PER_M = TRANSLATIONAL_DRAG_10MM_NS_PER_M
    CASE_LABEL = "10 mm thesis robot in glycerin"
elif ROBOT_CASE == "SCALED_100UM":
    MR_DIPOLE_MOMENT_AM2 = MR_DIPOLE_MOMENT_100UM_AM2
    TRANSLATIONAL_DRAG_NS_PER_M = TRANSLATIONAL_DRAG_100UM_NS_PER_M
    CASE_LABEL = "100 um scaled robot in blood"
else:
    raise ValueError("ROBOT_CASE must be 'THESIS_10MM' or 'SCALED_100UM'.")

SPEED_SCALE = 1e3  # m/s -> mm/s

# ============================================================
# HELPERS
# ============================================================
def shortest_angle_path_deg(theta_deg):
    theta_deg = np.asarray(theta_deg, dtype=float)
    out = np.zeros_like(theta_deg)
    out[0] = theta_deg[0]
    for i in range(1, len(theta_deg)):
        d = theta_deg[i] - theta_deg[i - 1]
        d = (d + 180.0) % 360.0 - 180.0
        out[i] = out[i - 1] + d
    return out


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        return None
    return v / n


def gradient_1d(y, x):
    return np.gradient(np.asarray(y, dtype=float), np.asarray(x, dtype=float), edge_order=2)


def smooth_for_label(y, window):
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < 3:
        return y.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, pad_width=pad, mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def add_curve_label(ax, x, y, text, line, case_key):
    cfg = LABELS[case_key][text]
    y_anchor = smooth_for_label(y, LABEL_SMOOTH_WINDOW)
    idx = int(np.clip(round(cfg.get("frac", 0.5) * (len(x) - 1)), 0, len(x) - 1))
    ax.annotate(
        text,
        xy=(x[idx], y_anchor[idx]),
        xytext=(cfg.get("dx", 8), cfg.get("dy", 0)),
        textcoords="offset points",
        color=line.get_color(),
        fontsize=cfg.get("fontsize", LABEL_FONT_SIZE),
        rotation=cfg.get("rotation", 0),
        ha=cfg.get("ha", "left"),
        va=cfg.get("va", "center"),
    )


def angle_signed_deg(u, v):
    cross_z = u[0] * v[1] - u[1] * v[0]
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.degrees(np.arctan2(cross_z, dot))


def load_path_csv(csv_file):
    path = Path(csv_file)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {csv_file}")

    df = pd.read_csv(path)
    required = {"theta_x_deg", "theta_y_deg", "x_pred_cm", "y_pred_cm"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{csv_file} is missing required columns: {sorted(missing)}")

    if "time_s" in df.columns:
        t_s = df["time_s"].to_numpy(dtype=float)
    elif "t_s" in df.columns:
        t_s = df["t_s"].to_numpy(dtype=float)
    elif "dt_s" in df.columns:
        dt = df["dt_s"].to_numpy(dtype=float)
        t_s = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    else:
        raise KeyError(f"{csv_file} needs time_s, t_s, or dt_s.")

    theta_x = shortest_angle_path_deg(df["theta_x_deg"].to_numpy(dtype=float))
    theta_y = shortest_angle_path_deg(df["theta_y_deg"].to_numpy(dtype=float))
    x_cm = df["x_pred_cm"].to_numpy(dtype=float)
    y_cm = df["y_pred_cm"].to_numpy(dtype=float)

    x_m = x_cm * 1e-2
    y_m = y_cm * 1e-2
    pts_m = np.column_stack([x_m, y_m, np.zeros_like(x_m)])

    step_ds_m = np.sqrt(np.diff(x_m) ** 2 + np.diff(y_m) ** 2)
    s_m = np.concatenate([[0.0], np.cumsum(step_ds_m)])
    s_mm = s_m * 1e3

    vx_path = gradient_1d(x_m, t_s)
    vy_path = gradient_1d(y_m, t_s)
    path_vel_xy = np.column_stack([vx_path, vy_path])

    return df, t_s, theta_x, theta_y, x_cm, y_cm, pts_m, s_mm, path_vel_xy


# Shared geometry objects
pool_center_m = np.array([POOL_SIZE_CM / 2.0, POOL_SIZE_CM / 2.0, 0.0]) * 1e-2
top_pos_m = pool_center_m + np.array([0.0, 0.0, +MAGNET_OFFSET_Z_CM]) * 1e-2
bot_pos_m = pool_center_m + np.array([0.0, 0.0, -MAGNET_OFFSET_Z_CM]) * 1e-2
single_mag_pos_m = bot_pos_m.copy()
ex = np.array([H, 0.0, 0.0])
ey = np.array([0.0, H, 0.0])
ez = np.array([0.0, 0.0, H])


def angles_to_mhat_sync(theta_x_deg, theta_y_deg):
    tx = np.deg2rad(theta_x_deg)
    ty = np.deg2rad(theta_y_deg)
    beta = np.hypot(tx, ty)
    if beta < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    s = np.sin(beta)
    c = np.cos(beta)
    m_hat = np.array([(ty / beta) * s, -(tx / beta) * s, c], dtype=float)
    return m_hat / np.linalg.norm(m_hat)


def make_single_source(ax_deg, ay_deg):
    m_hat = angles_to_mhat_sync(ax_deg, ay_deg)
    return magpy.magnet.Cuboid(
        polarization=tuple(BR_T * m_hat),
        dimension=(MAGNET_SIDE_CM * 1e-2, MAGNET_SIDE_CM * 1e-2, MAGNET_SIDE_CM * 1e-2),
        position=tuple(single_mag_pos_m),
    )


def make_double_sources(ax_deg, ay_deg):
    mag_top = magpy.magnet.Cuboid(
        polarization=(0, 0, BR_T),
        dimension=(MAGNET_SIDE_CM * 1e-2, MAGNET_SIDE_CM * 1e-2, MAGNET_SIDE_CM * 1e-2),
        position=tuple(top_pos_m),
    )
    mag_bot = magpy.magnet.Cuboid(
        polarization=(0, 0, BR_T),
        dimension=(MAGNET_SIDE_CM * 1e-2, MAGNET_SIDE_CM * 1e-2, MAGNET_SIDE_CM * 1e-2),
        position=tuple(bot_pos_m),
    )
    mag_top.rotate_from_angax(angle=ax_deg, axis=[1, 0, 0], anchor=tuple(top_pos_m))
    mag_bot.rotate_from_angax(angle=ay_deg, axis=[0, 1, 0], anchor=tuple(bot_pos_m))
    return [mag_top, mag_bot]


def getB_sum(sources, points_m):
    B = magpy.getB(sources, points_m)
    B = np.asarray(B, dtype=float)
    if B.ndim == 3:
        B = B.sum(axis=0)
    elif B.ndim == 2 and isinstance(sources, (list, tuple)) and B.shape[0] == len(sources) and B.shape[1] == 3:
        B = B.sum(axis=0)
    return B


def field_and_force_at_point(sources, pos_m, dipole_moment_am2):
    B0 = np.asarray(getB_sum(sources, pos_m.reshape(1, 3))).reshape(3)
    Bnorm = np.linalg.norm(B0)
    if Bnorm < 1e-15:
        return np.zeros(3), B0

    m_hat_robot = B0 / Bnorm

    Bxp = np.asarray(getB_sum(sources, (pos_m + ex).reshape(1, 3))).reshape(3)
    Bxm = np.asarray(getB_sum(sources, (pos_m - ex).reshape(1, 3))).reshape(3)
    Byp = np.asarray(getB_sum(sources, (pos_m + ey).reshape(1, 3))).reshape(3)
    Bym = np.asarray(getB_sum(sources, (pos_m - ey).reshape(1, 3))).reshape(3)
    Bzp = np.asarray(getB_sum(sources, (pos_m + ez).reshape(1, 3))).reshape(3)
    Bzm = np.asarray(getB_sum(sources, (pos_m - ez).reshape(1, 3))).reshape(3)

    dBdx = (Bxp - Bxm) / (2.0 * H)
    dBdy = (Byp - Bym) / (2.0 * H)
    dBdz = (Bzp - Bzm) / (2.0 * H)

    Fx = dipole_moment_am2 * np.dot(dBdx, m_hat_robot)
    Fy = dipole_moment_am2 * np.dot(dBdy, m_hat_robot)
    Fz = dipole_moment_am2 * np.dot(dBdz, m_hat_robot)
    return np.array([Fx, Fy, Fz]), B0


def compute_case(csv_file, case_key):
    df, t_s, theta_x, theta_y, x_cm, y_cm, pts_m, s_mm, path_vel_xy = load_path_csv(csv_file)

    F_xyz = np.zeros((len(df), 3), dtype=float)
    B_xyz = np.zeros((len(df), 3), dtype=float)

    for i, (ax_deg, ay_deg, pos_m) in enumerate(zip(theta_x, theta_y, pts_m)):
        if case_key == "single":
            sources = make_single_source(ax_deg, ay_deg)
        elif case_key == "double":
            sources = make_double_sources(ax_deg, ay_deg)
        else:
            raise ValueError("case_key must be 'single' or 'double'.")
        F_xyz[i], B_xyz[i] = field_and_force_at_point(sources, pos_m, MR_DIPOLE_MOMENT_AM2)

    v_grad_xy_mps_vec = F_xyz[:, :2] / TRANSLATIONAL_DRAG_NS_PER_M
    v_grad_mag_mm_s = np.linalg.norm(v_grad_xy_mps_vec, axis=1) * SPEED_SCALE

    that_xy = np.zeros_like(v_grad_xy_mps_vec)
    nhat_xy = np.zeros_like(v_grad_xy_mps_vec)
    angle_signed = np.full(len(df), np.nan)

    for i, vel in enumerate(path_vel_xy):
        t_hat = normalize(vel)
        if t_hat is None:
            if i > 0:
                that_xy[i] = that_xy[i - 1]
                nhat_xy[i] = nhat_xy[i - 1]
            continue

        that_xy[i] = t_hat
        nhat_xy[i] = np.array([-t_hat[1], t_hat[0]])

        g_hat = normalize(v_grad_xy_mps_vec[i])
        if g_hat is not None:
            angle_signed[i] = angle_signed_deg(t_hat, g_hat)

    for arr in (that_xy, nhat_xy):
        for i in range(1, len(arr)):
            if np.linalg.norm(arr[i]) < 1e-15:
                arr[i] = arr[i - 1]

    v_parallel_mm_s = np.sum(v_grad_xy_mps_vec * that_xy, axis=1) * SPEED_SCALE
    v_perp_mm_s = np.sum(v_grad_xy_mps_vec * nhat_xy, axis=1) * SPEED_SCALE
    angle_abs = np.abs(angle_signed)

    # Fill possible leading NaNs caused by repeated first points.
    for arr in (angle_signed, angle_abs):
        valid = np.where(np.isfinite(arr))[0]
        if len(valid) and valid[0] > 0:
            arr[:valid[0]] = arr[valid[0]]

    out = pd.DataFrame({
        "time_s": t_s,
        "arc_length_mm": s_mm,
        "x_pred_cm": x_cm,
        "y_pred_cm": y_cm,
        "Bx_mT": B_xyz[:, 0] * 1e3,
        "By_mT": B_xyz[:, 1] * 1e3,
        "Bz_mT": B_xyz[:, 2] * 1e3,
        "Fx_uN": F_xyz[:, 0] * 1e6,
        "Fy_uN": F_xyz[:, 1] * 1e6,
        "Fz_uN": F_xyz[:, 2] * 1e6,
        "gradient_speed_mm_s": v_grad_mag_mm_s,
        "v_parallel_mm_s": v_parallel_mm_s,
        "v_perp_mm_s": v_perp_mm_s,
        "angle_error_deg_signed": angle_signed,
        "angle_error_deg_abs": angle_abs,
    })

    return {
        "case_key": case_key,
        "s_mm": s_mm,
        "v_mag": v_grad_mag_mm_s,
        "angle_abs": angle_abs,
        "v_parallel": v_parallel_mm_s,
        "v_perp": v_perp_mm_s,
        "table": out,
    }


def finite_limits(*arrays, pad_frac=0.05, force_bottom=None, include_zero=False):
    vals = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if force_bottom is not None:
        lo = force_bottom
    if np.isclose(lo, hi):
        pad = 1.0 if np.isclose(lo, 0.0) else abs(lo) * 0.1
    else:
        pad = (hi - lo) * pad_frac
    return (lo - (0 if force_bottom is not None else pad), hi + pad)


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=FONT_SIZE)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.margins(x=0.03, y=0.10)


# ============================================================
# RUN CALCULATIONS
# ============================================================
single = compute_case(CSV_SINGLE, "single")
double = compute_case(CSV_DOUBLE, "double")

single["table"].to_csv(OUT_SINGLE_CSV, index=False)
double["table"].to_csv(OUT_DOUBLE_CSV, index=False)

# Shared axis limits per column, so single and double are directly comparable.
xlim_all = finite_limits(single["s_mm"], double["s_mm"], pad_frac=0.03, force_bottom=0.0)
ylim_mag = finite_limits(single["v_mag"], double["v_mag"], pad_frac=0.08, force_bottom=0.0)
ylim_ang = finite_limits(single["angle_abs"], double["angle_abs"], pad_frac=0.08, force_bottom=0.0)
ylim_comp = finite_limits(
    single["v_parallel"], single["v_perp"], double["v_parallel"], double["v_perp"],
    pad_frac=0.08, include_zero=True
)

# ============================================================
# PLOT: 2 rows x 3 columns
# ============================================================
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=FIGSIZE, sharex="col")

cases = [(single, "Single PMAS"), (double, "Double PMAS")]
for row, (data, row_title) in enumerate(cases):
    x = data["s_mm"]
    case_key = data["case_key"]

    # Column 1: magnitude
    ax = axes[row, 0]
    line_mag, = ax.plot(x, data["v_mag"], lw=LW)
    ax.set_ylim(*ylim_mag)
    ax.set_xlim(*xlim_all)
    ax.set_ylabel(f"{row_title}\nVelocity / mm/s", fontsize=FONT_SIZE)
    #add_curve_label(ax, x, data["v_mag"], "gradient speed magnitude", line_mag, case_key)

    # Column 2: angle
    ax = axes[row, 1]
    line_ang, = ax.plot(x, data["angle_abs"], lw=LW)
    ax.set_ylim(*ylim_ang)
    ax.set_xlim(*xlim_all)
    ax.set_ylabel("Angle / deg", fontsize=FONT_SIZE)
    #add_curve_label(ax, x, data["angle_abs"], "absolute direction error", line_ang, case_key)

    # Column 3: components
    ax = axes[row, 2]
    line_par, = ax.plot(x, data["v_parallel"], lw=LW)
    line_perp, = ax.plot(x, data["v_perp"], lw=LW)
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.set_ylim(*ylim_comp)
    ax.set_xlim(*xlim_all)
    ax.set_ylabel("Velocity / mm/s", fontsize=FONT_SIZE)
    add_curve_label(ax, x, data["v_parallel"], "along-path", line_par, case_key)
    add_curve_label(ax, x, data["v_perp"], "lateral", line_perp, case_key)

for ax in axes.ravel():
    style_axis(ax)

axes[0, 0].set_title("Translational Velocity Magnitude", fontsize=TITLE_FONT_SIZE)
axes[0, 1].set_title("Absolute Direction Error", fontsize=TITLE_FONT_SIZE)
axes[0, 2].set_title("Translational Velocity Components", fontsize=TITLE_FONT_SIZE)

for ax in axes[1, :]:
    ax.set_xlabel("Arc length along path / mm", fontsize=FONT_SIZE)

fig.tight_layout()
plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
plt.show()

print("Saved:")
print(f"  - {OUT_PNG}")
print(f"  - {OUT_SINGLE_CSV}")
print(f"  - {OUT_DOUBLE_CSV}")
print()
print("Model summary:")
print(f"  Robot case                     = {CASE_LABEL}")
print(f"  Dipole moment                  = {MR_DIPOLE_MOMENT_AM2:.6e} A·m^2")
print(f"  Translational drag coefficient = {TRANSLATIONAL_DRAG_NS_PER_M:.6e} N·s/m")
print("  Rows                           = single magnet, double magnet")
print("  Columns                        = magnitude, angle, components")
print("  Axis sharing                   = each column uses identical y-limits for both rows")
