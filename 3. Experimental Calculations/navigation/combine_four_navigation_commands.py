#!/usr/bin/env python3
"""
combine_four_navigation_commands.py

Use this after processing the four navigation commands with the manual ghost-sample script.

Expected structure
------------------
Put each command result in its own folder, for example:

all_nav_results/
    cmd_1/
        DATA_clicked_centers_positions_velocity.csv
        DATA_clicked_segment_velocity.csv
        DATA_raw_manual_samples_px.csv
        analysis_settings.csv
        FIGURE_warped_oriented_first_frame.png

    cmd_2/
        ...

    cmd_3/
        ...

    cmd_4/
        ...

Then run:
    python combine_four_navigation_commands.py --root all_nav_results --frequency-hz 0.5

Outputs
-------
In all_nav_results/combined_analysis/:

1. FIGURE_combined_cross_paths.png
   - all desired paths and measured paths in one common-origin plot

2. FIGURE_combined_cross_paths_clean.png
   - same, but no background image

3. COMBINED_all_segments.csv
   - all segment velocities together

4. COMBINED_all_points.csv
   - all clicked point positions together

5. COMBINED_summary_by_command.csv
   - average velocity/force/displacement per command

6. COMBINED_summary_overall.csv
   - overall average values

Meaning
-------
v_parallel:
    velocity along desired command direction

v_perp:
    velocity to the right-hand perpendicular direction if your input CSV
    already uses your corrected sign convention.

F_eff:
    equivalent effective force = zeta * v

phase displacement:
    ds/dphi = v / omega
    ds per rotation = v / f
"""

from pathlib import Path
import argparse
import math

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Drag coefficient
# ============================================================

def perrin_translational_drag_ns_per_m(eta_pa_s=1.14, a_m=5e-3, b_m=3.25e-3):
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


# ============================================================
# Helpers
# ============================================================

def find_command_folders(root: Path):
    folders = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "DATA_clicked_segment_velocity.csv").exists() and (p / "DATA_clicked_centers_positions_velocity.csv").exists():
            folders.append(p)
    return folders


def read_settings(folder: Path):
    settings_path = folder / "analysis_settings.csv"
    if not settings_path.exists():
        return {}
    df = pd.read_csv(settings_path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def choose_background(command_folders):
    """
    Use the first available warped/oriented first frame as background.
    """
    names = [
        "FIGURE_warped_oriented_first_frame.png",
        "warped_oriented_first_frame.png",
        "FIGURE_final_overlay.png",
    ]
    for folder in command_folders:
        for name in names:
            path = folder / name
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    return img
    return None


def mm_to_px(x_mm, y_mm, origin_px, px_per_mm):
    ox, oy = origin_px
    return np.array([ox + x_mm * px_per_mm, oy - y_mm * px_per_mm], dtype=np.float32)


def draw_text(img, lines, x=10, y=25):
    out = img.copy()
    for line in lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3)
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1)
        y += 24
    return out


def add_effective_columns(seg, frequency_hz, zeta_ns_per_m):
    seg = seg.copy()
    omega = 2.0 * np.pi * frequency_hz

    # Phase
    if "time_mid_s" in seg.columns:
        seg["phase_rad"] = (omega * seg["time_mid_s"].astype(float)) % (2.0 * np.pi)
        seg["phase_deg"] = np.rad2deg(seg["phase_rad"])

    # Effective force, v in mm/s -> m/s
    for name, col in [
        ("parallel", "v_parallel_mm_s"),
        ("perp", "v_perp_mm_s"),
        ("speed", "speed_mm_s"),
    ]:
        if col in seg.columns:
            seg[f"F_eff_{name}_N"] = zeta_ns_per_m * seg[col].astype(float) * 1e-3
            seg[f"F_eff_{name}_uN"] = seg[f"F_eff_{name}_N"] * 1e6
            seg[f"ds_{name}_mm_per_rad"] = seg[col].astype(float) / omega
            seg[f"ds_{name}_mm_per_rotation"] = seg[col].astype(float) / frequency_hz

    return seg


def summarize_group(df, group_name):
    rows = []
    cols = [
        "v_parallel_mm_s",
        "v_perp_mm_s",
        "speed_mm_s",
        "F_eff_parallel_uN",
        "F_eff_perp_uN",
        "F_eff_speed_uN",
        "ds_parallel_mm_per_rad",
        "ds_perp_mm_per_rad",
        "ds_speed_mm_per_rad",
        "ds_parallel_mm_per_rotation",
        "ds_perp_mm_per_rotation",
        "ds_speed_mm_per_rotation",
    ]

    for col in cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue

        rows.append({
            "group": group_name,
            "quantity": col,
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
            "n": len(values),
        })

    # signed and absolute drift ratios
    if "v_parallel_mm_s" in df.columns and "v_perp_mm_s" in df.columns:
        vp = pd.to_numeric(df["v_parallel_mm_s"], errors="coerce")
        vn = pd.to_numeric(df["v_perp_mm_s"], errors="coerce")
        mask = vp.notna() & vn.notna() & (vp.abs() > 1e-12)
        if mask.any():
            ratio = vn[mask] / vp[mask]
            rows.append({
                "group": group_name,
                "quantity": "v_perp_over_v_parallel",
                "mean": ratio.mean(),
                "std": ratio.std(),
                "median": ratio.median(),
                "min": ratio.min(),
                "max": ratio.max(),
                "n": len(ratio),
            })
            rows.append({
                "group": group_name,
                "quantity": "abs_v_perp_over_abs_v_parallel",
                "mean": ratio.abs().mean(),
                "std": ratio.abs().std(),
                "median": ratio.abs().median(),
                "min": ratio.abs().min(),
                "max": ratio.abs().max(),
                "n": len(ratio),
            })

    return rows


def make_blank_canvas(width=1200, height=1200):
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    # grid
    spacing = 80
    for x in range(0, width, spacing):
        cv2.line(img, (x, 0), (x, height), (210, 210, 210), 1)
    for y in range(0, height, spacing):
        cv2.line(img, (0, y), (width, y), (210, 210, 210), 1)
    return img


def draw_combined_figure(command_data, background, output_path, clean_output_path):
    """
    command_data list entries:
       command_name, points df, settings dict
    """
    # Determine coordinate system from first settings.
    first_settings = command_data[0]["settings"]
    px_per_mm = float(first_settings.get("px_per_mm", 8.0))
    origin_px = np.array([
        float(first_settings.get("origin_px_x", 600.0)),
        float(first_settings.get("origin_px_y", 600.0)),
    ], dtype=np.float32)

    if background is None:
        canvas = make_blank_canvas()
        # Center the origin on the blank canvas and redraw paths relative to it.
        origin_px = np.array([canvas.shape[1] / 2, canvas.shape[0] / 2], dtype=np.float32)
    else:
        canvas = background.copy()

    clean = make_blank_canvas(width=canvas.shape[1], height=canvas.shape[0])

    # Use same origin in clean canvas as image canvas.
    clean_origin = origin_px.copy()

    colors = [
        (255, 255, 0),   # cyan measured
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
    ]

    for i, entry in enumerate(command_data):
        name = entry["name"]
        points = entry["points"]
        settings = entry["settings"]
        color = colors[i % len(colors)]

        if points.empty:
            continue

        # Measured path from x_mm/y_mm.
        measured_px = []
        for _, r in points.iterrows():
            measured_px.append(mm_to_px(float(r["x_mm"]), float(r["y_mm"]), origin_px, px_per_mm))
        measured_px = np.array(measured_px, dtype=np.float32)

        # Desired path from settings, if present.
        x0 = float(settings.get("desired_start_x_mm", points["x_mm"].iloc[0]))
        y0 = float(settings.get("desired_start_y_mm", points["y_mm"].iloc[0]))
        x1 = float(settings.get("desired_end_x_mm", points["x_mm"].iloc[-1]))
        y1 = float(settings.get("desired_end_y_mm", points["y_mm"].iloc[-1]))

        d0 = mm_to_px(x0, y0, origin_px, px_per_mm)
        d1 = mm_to_px(x1, y1, origin_px, px_per_mm)

        for img in [canvas, clean]:
            # Desired paths yellow.
            cv2.line(img, tuple(d0.astype(int)), tuple(d1.astype(int)), color, 2)
            cv2.circle(img, tuple(d1.astype(int)), 5, (0, 0, 255), -1)

            # Measured trajectory:
            # - use sampled points as red dots
            # - fit a linear line through all points EXCEPT the last one
            #   because the last point can correspond to the robot no longer rotating significantly.
            if len(measured_px) > 1:
                # Fit includes the origin, then all sampled points except the last one.
                # The last sample is excluded because the robot may no longer be rotating significantly.
                if len(measured_px) > 2:
                    fit_px = np.vstack([origin_px.reshape(1, 2), measured_px[:-1]])
                else:
                    fit_px = np.vstack([origin_px.reshape(1, 2), measured_px])

                x = fit_px[:, 0]
                y = fit_px[:, 1]

                # Fit in the numerically stable direction.
                # If path is mostly horizontal: y = a*x + b.
                # If path is mostly vertical: x = a*y + b.
                if np.std(x) >= np.std(y):
                    coeff = np.polyfit(x, y, 1)
                    # Draw from origin to the furthest included point in x.
                    x_fit = np.linspace(x.min(), x.max(), 300)
                    y_fit = coeff[0] * x_fit + coeff[1]
                else:
                    coeff = np.polyfit(y, x, 1)
                    # Draw from origin to the furthest included point in y.
                    y_fit = np.linspace(y.min(), y.max(), 300)
                    x_fit = coeff[0] * y_fit + coeff[1]

                fit_pts = np.column_stack([x_fit, y_fit]).astype(np.int32)

                # Draw a thick black outline first, then the colored fitted line.
                # This keeps the line visible on the photographic/background image.
                cv2.polylines(img, [fit_pts], False, (0, 0, 0), 8)
                cv2.polylines(img, [fit_pts], False, color, 4)

            # Sampled clicked points: red with black outline.
            for p in measured_px:
                pi = tuple(p.astype(int))
                cv2.circle(img, pi, 7, (0, 0, 0), -1)
                cv2.circle(img, pi, 5, (0, 0, 255), -1)

            # From origin to first sample, also outlined.
            #if len(measured_px) > 0:
            #    p0 = tuple(measured_px[0].astype(int))
            #    cv2.line(img, tuple(origin_px.astype(int)), p0, (0, 0, 0), 6)
            #    cv2.line(img, tuple(origin_px.astype(int)), p0, color, 3)

            # Label near end
            if len(measured_px) > 0:
                endp = measured_px[-1].astype(int)
                label_offsets = {
                    "results_bl": (-50, -10),
                    "results_br": (-20, 20),
                    "results_tl": (-50, -10),
                    "results_tr": (10, -10),
                }

                dx, dy = label_offsets.get(name, (8, 8))

                cv2.putText(
                    img,
                    name,
                    (int(endp[0] + dx), int(endp[1] + dy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

    for img in [canvas, clean]:
        cv2.drawMarker(img, tuple(origin_px.astype(int)), (255, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
        cv2.putText(img, "origin", tuple((origin_px + np.array([70, -35])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        cv2.arrowedLine(img, tuple(origin_px.astype(int)), tuple((origin_px + np.array([50, 0])).astype(int)),
                        (255, 0, 255), 2)
        cv2.putText(img, "+x", tuple((origin_px + np.array([55, 5])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        cv2.arrowedLine(img, tuple(origin_px.astype(int)), tuple((origin_px + np.array([0, -50])).astype(int)),
                        (255, 0, 255), 2)
        cv2.putText(img, "+y", tuple((origin_px + np.array([-10, -55])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    #canvas = draw_text(canvas, [
    #    "yellow = commanded directions",
    #    "colored lines = linear fits including origin, red dots = sampled points",
    #])
    #clean = draw_text(clean, [
    #    "yellow = commanded directions",
    #    "colored lines = linear fits including origin, red dots = sampled points",
    #])

    cv2.imwrite(str(output_path), canvas)
    cv2.imwrite(str(clean_output_path), clean)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing one subfolder per command")
    ap.add_argument("--frequency-hz", type=float, default=0.5)
    ap.add_argument("--zeta", type=float, default=None, help="Resistance coefficient in N s/m")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / "combined_analysis"
    out_dir.mkdir(exist_ok=True)

    zeta = args.zeta
    if zeta is None:
        zeta = perrin_translational_drag_ns_per_m()

    command_folders = find_command_folders(root)
    if not command_folders:
        raise RuntimeError(
            f"No command folders found in {root}. Each subfolder must contain "
            "DATA_clicked_segment_velocity.csv and DATA_clicked_centers_positions_velocity.csv"
        )

    command_data = []
    all_seg = []
    all_points = []
    by_command_rows = []

    for i, folder in enumerate(command_folders, start=1):
        name = folder.name
        settings = read_settings(folder)

        seg = pd.read_csv(folder / "DATA_clicked_segment_velocity.csv")
        pts = pd.read_csv(folder / "DATA_clicked_centers_positions_velocity.csv")

        seg = add_effective_columns(seg, args.frequency_hz, zeta)
        seg["command"] = name
        pts["command"] = name

        all_seg.append(seg)
        all_points.append(pts)

        by_command_rows.extend(summarize_group(seg, name))

        command_data.append({
            "name": name,
            "settings": settings,
            "segments": seg,
            "points": pts,
        })

    all_seg_df = pd.concat(all_seg, ignore_index=True)
    all_points_df = pd.concat(all_points, ignore_index=True)

    by_command_summary = pd.DataFrame(by_command_rows)
    overall_summary = pd.DataFrame(summarize_group(all_seg_df, "overall"))

    all_seg_df.to_csv(out_dir / "COMBINED_all_segments.csv", index=False)
    all_points_df.to_csv(out_dir / "COMBINED_all_points.csv", index=False)
    by_command_summary.to_csv(out_dir / "COMBINED_summary_by_command.csv", index=False)
    overall_summary.to_csv(out_dir / "COMBINED_summary_overall.csv", index=False)

    background = choose_background(command_folders)
    draw_combined_figure(
        command_data,
        background=background,
        output_path=out_dir / "FIGURE_combined_cross_paths.png",
        clean_output_path=out_dir / "FIGURE_combined_cross_paths_clean.png",
    )

    print("\nDone.")
    print(f"Commands found: {len(command_folders)}")
    print(f"frequency_hz = {args.frequency_hz}")
    print(f"zeta_N_s_per_m = {zeta:.6e}")
    print("\nSaved to:")
    print(out_dir)
    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))


if __name__ == "__main__":
    main()
