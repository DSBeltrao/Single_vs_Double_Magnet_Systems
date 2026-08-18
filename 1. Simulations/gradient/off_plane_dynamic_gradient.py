import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator

# ============================================================
# INPUT
# ============================================================
CSV_FILE = "gradient_z_speed_compare_single_vs_double.csv"

OUT_PNG = "z_gradient_vs_gravity_compare.png"
OUT_CSV = "z_gradient_vs_gravity_compare.csv"

# ============================================================
# CHOOSE WHICH ROBOT / FLUID CASE YOU WANT
# ============================================================
CASE_NAME = "10mm robot in glycerin"
FONT_SIZE = 20

# ------------------------------------------------------------
# Effective gravity:
#   Fg_eff = (rho_robot - rho_fluid) * V * g
#
# Fill these in with the values you want to test.
# Units:
#   rho_robot  [kg/m^3]
#   rho_fluid  [kg/m^3]
#   V_robot    [m^3]
# ------------------------------------------------------------
RHO_ROBOT_KG_M3 = 7500.0      # EXAMPLE placeholder
RHO_FLUID_KG_M3 = 1260.0      # glycerin example
V_ROBOT_M3      = 2.36e-9     # replace if your full robot volume differs
G_M_S2          = 9.81

LABEL_SMOOTH_WINDOW = 15

def smooth_for_label(y, window):
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < 3:
        return y.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    pad = window // 2
    ypad = np.pad(y, pad, mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def direct_label(ax, x, y_anchor, text, color, frac, dx=8, dy=0):
    idx = int(np.clip(round(frac * (len(x) - 1)), 0, len(x) - 1))
    ax.annotate(
        text,
        xy=(x[idx], y_anchor[idx]),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=FONT_SIZE,
        ha="left",
        va="center",
    )

# ============================================================
# SAME TRANSLATIONAL DRAG MODEL AS BEFORE
# ============================================================
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

# ------------------------------------------------------------
# Match this to the case you want
# ------------------------------------------------------------
ROBOT_CASE = "THESIS_10MM"   # "THESIS_10MM" or "SCALED_100UM"

SCALE_LAMBDA = 0.01

ETA_GLYCERIN_PA_S = 1.14
A_ELLIP_10MM_M = 5.0e-3
B_ELLIP_10MM_M = 3.25e-3

ETA_BLOOD_PA_S = 5e-3
A_ELLIP_100UM_M = A_ELLIP_10MM_M * SCALE_LAMBDA
B_ELLIP_100UM_M = B_ELLIP_10MM_M * SCALE_LAMBDA

TRANSLATIONAL_DRAG_10MM_NS_PER_M = perrin_translational_drag_ns_per_m(
    ETA_GLYCERIN_PA_S, A_ELLIP_10MM_M, B_ELLIP_10MM_M
)

TRANSLATIONAL_DRAG_100UM_NS_PER_M = perrin_translational_drag_ns_per_m(
    ETA_BLOOD_PA_S, A_ELLIP_100UM_M, B_ELLIP_100UM_M
)

if ROBOT_CASE == "THESIS_10MM":
    TRANSLATIONAL_DRAG_NS_PER_M = TRANSLATIONAL_DRAG_10MM_NS_PER_M
elif ROBOT_CASE == "SCALED_100UM":
    TRANSLATIONAL_DRAG_NS_PER_M = TRANSLATIONAL_DRAG_100UM_NS_PER_M
else:
    raise ValueError("ROBOT_CASE must be 'THESIS_10MM' or 'SCALED_100UM'.")

# ============================================================
# LOAD CSV
# ============================================================
path = Path(CSV_FILE)
if not path.exists():
    raise FileNotFoundError(f"Could not find {CSV_FILE}")

df = pd.read_csv(path)

required = {
    "arc_length_single_mm", "vz_single_mm_s",
    "arc_length_double_mm", "vz_double_mm_s"
}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns in {CSV_FILE}: {sorted(missing)}")

arc_single_mm = df["arc_length_single_mm"].to_numpy(dtype=float)
vz_single_mm_s = df["vz_single_mm_s"].to_numpy(dtype=float)

arc_double_mm = df["arc_length_double_mm"].to_numpy(dtype=float)
vz_double_mm_s = df["vz_double_mm_s"].to_numpy(dtype=float)

# remove NaN padding if present
mask_single = np.isfinite(arc_single_mm) & np.isfinite(vz_single_mm_s)
mask_double = np.isfinite(arc_double_mm) & np.isfinite(vz_double_mm_s)

arc_single_mm = arc_single_mm[mask_single]
vz_single_mm_s = vz_single_mm_s[mask_single]

arc_double_mm = arc_double_mm[mask_double]
vz_double_mm_s = vz_double_mm_s[mask_double]

# ============================================================
# GRAVITY / BUOYANCY
# ============================================================
delta_rho = RHO_ROBOT_KG_M3 - RHO_FLUID_KG_M3
Fg_eff_N = delta_rho * V_ROBOT_M3 * G_M_S2

# Sign convention:
# positive z speed = upward
# gravity points downward, so:
vz_gravity_m_s = -Fg_eff_N / TRANSLATIONAL_DRAG_NS_PER_M
vz_gravity_mm_s = vz_gravity_m_s * 1e3

# Net vertical speeds
vz_net_single_mm_s = vz_single_mm_s + vz_gravity_mm_s
vz_net_double_mm_s = vz_double_mm_s + vz_gravity_mm_s

# ============================================================
# SAVE OUTPUT TABLE
# ============================================================
nmax = max(len(arc_single_mm), len(arc_double_mm))

def pad(arr, n):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == n:
        return arr
    out = np.full(n, np.nan, dtype=float)
    out[:len(arr)] = arr
    return out

out = pd.DataFrame({
    "arc_length_single_mm": pad(arc_single_mm, nmax),
    "vz_single_mag_mm_s": pad(vz_single_mm_s, nmax),
    "vz_single_net_mm_s": pad(vz_net_single_mm_s, nmax),

    "arc_length_double_mm": pad(arc_double_mm, nmax),
    "vz_double_mag_mm_s": pad(vz_double_mm_s, nmax),
    "vz_double_net_mm_s": pad(vz_net_double_mm_s, nmax),

    "vz_gravity_mm_s": np.full(nmax, vz_gravity_mm_s),
    "Fg_eff_N": np.full(nmax, Fg_eff_N),
    "rho_robot_kg_m3": np.full(nmax, RHO_ROBOT_KG_M3),
    "rho_fluid_kg_m3": np.full(nmax, RHO_FLUID_KG_M3),
    "V_robot_m3": np.full(nmax, V_ROBOT_M3),
})
out.to_csv(OUT_CSV, index=False)

# ============================================================
# PLOT 1: magnetic z-speed vs gravity
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 5.2))

# Plot lines (NO labels here)
line_single, = ax.plot(arc_single_mm, vz_single_mm_s, lw=2)
line_double, = ax.plot(arc_double_mm, vz_double_mm_s, lw=2)
line_grav = ax.axhline(vz_gravity_mm_s, lw=2)
ax.axhline(0.0, color="0.4", lw=1.0)

# Smooth for nicer label placement
y_single_anchor = smooth_for_label(vz_single_mm_s, LABEL_SMOOTH_WINDOW)
y_double_anchor = smooth_for_label(vz_double_mm_s, LABEL_SMOOTH_WINDOW)
y_grav_anchor = np.full_like(arc_single_mm, vz_gravity_mm_s)

# Direct labels (tuned positions)
direct_label(ax, arc_single_mm, y_single_anchor,
             "single magnet: magnetic z-speed",
             line_single.get_color(), 0.65, dy=20)

direct_label(ax, arc_double_mm, y_double_anchor,
             "double magnet: magnetic z-speed",
             line_double.get_color(), 0.35, dy=-20)

direct_label(ax, arc_single_mm, y_grav_anchor,
             "gravity settling speed",
             line_grav.get_color(), 0.75, dy=10)

# Axes
ax.set_xlabel("Arc length along path / mm")
ax.set_ylabel("Vertical speed / mm/s")

# Styling (same as your other script)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# PLOT 2: net z-speed
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 5.2))

line_single, = ax.plot(arc_single_mm, vz_net_single_mm_s, lw=2)
line_double, = ax.plot(arc_double_mm, vz_net_double_mm_s, lw=2)
ax.axhline(0.0, color="0.4", lw=1.0)

# Smooth anchors
y_single_anchor = smooth_for_label(vz_net_single_mm_s, LABEL_SMOOTH_WINDOW)
y_double_anchor = smooth_for_label(vz_net_double_mm_s, LABEL_SMOOTH_WINDOW)

# Labels
direct_label(ax, arc_single_mm, y_single_anchor,
             "single PMAS",
             line_single.get_color(), 0.45, dx = -35, dy=65)

direct_label(ax, arc_double_mm, y_double_anchor,
             "double PMAS",
             line_double.get_color(), 0.7, dx = 0, dy=-150)

# Axes
ax.set_xlabel("Arc length along path / mm", fontsize=FONT_SIZE)
ax.set_ylabel("Net vertical speed / mm/s", fontsize=FONT_SIZE)

ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=FONT_SIZE)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.tight_layout()
plt.savefig("z_net_speed_compare_single_vs_double.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# PRINT SUMMARY
# ============================================================
print("Saved:")
print(f"  - {OUT_CSV}")
print(f"  - {OUT_PNG}")
print("  - z_net_speed_compare_single_vs_double.png")
print()
print("Case:")
print(f"  {CASE_NAME}")
print(f"  rho_robot      = {RHO_ROBOT_KG_M3:.3f} kg/m^3")
print(f"  rho_fluid      = {RHO_FLUID_KG_M3:.3f} kg/m^3")
print(f"  V_robot        = {V_ROBOT_M3:.6e} m^3")
print(f"  Fg_eff         = {Fg_eff_N:.6e} N")
print(f"  vz_gravity     = {vz_gravity_mm_s:.6f} mm/s")
print()
print("Interpretation:")
print("  net > 0 : upward")
print("  net < 0 : downward")
print("  net = 0 : float / neutrally suspended in z")