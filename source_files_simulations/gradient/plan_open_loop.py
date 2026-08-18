"""
plan_open_loop.py

Sequential open-loop planner for a magnetic microrobot in a 2D pool.

Data flow
---------
1. Query the reference-path geometry at the current predicted arc length s.
2. Build the desired magnetic-field direction from the local Bishop frame.
3. Solve the inverse field problem for the two permanent magnets.
4. Use the chosen field magnitude to scale the predicted forward progress.
5. Advance the internal state and store one command row.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from planner_geometry import PathGeometry
from field_inverse_solver import FieldInverseSolver


COMMAND_RATE_HZ = 20.0
DT_S = 1.0 / COMMAND_RATE_HZ

ROTATION_FREQ_HZ = 1.0
PITCH_CM_PER_ROT = 1.0 / 3.0

DELTA_PHI_RAD = 2.0 * np.pi * ROTATION_FREQ_HZ * DT_S

MAX_TIME_S = 300.0
MAX_STEPS = int(MAX_TIME_S * COMMAND_RATE_HZ)

OUTPUT_CSV = "commands_open_loop.csv"


def desired_field_direction(N_hat: np.ndarray, B_hat: np.ndarray, phi_rad: float) -> np.ndarray:
    return np.cos(phi_rad) * N_hat + np.sin(phi_rad) * B_hat


def write_commands_csv(commands: list[dict], output_csv: str) -> None:
    if not commands:
        return

    fieldnames = list(commands[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(commands)


def plan_open_loop(
    waypoints_cm,
    s_start_cm: float = 0.0,
    phi_start_deg: float = 0.0,
    output_csv: str = OUTPUT_CSV,
) -> list[dict]:
    geometry = PathGeometry(waypoints_cm)

    solver = FieldInverseSolver(
        pool_size_cm=15.0,
        magnet_offset_z_cm=12.5,
        dipole_moment_Am2=66.0,
        cone_default_deg=5.0,
        cone_max_deg=12.0,
        dphi_deg=20.0,
        phase_sign=-1.0,
        B_expand_mT=6.0,
        B_sync_mT=4.0,
        B_min_mT=1.0,
        local_span_deg=30.0,
        local_step_deg=15.0,
        global_seeds_deg=(0.0, 90.0, 180.0, 270.0),
        w_angle=1.0,
        w_mag=2.0,
        w_align=0.05,
    )

    s_pred_cm = float(s_start_cm)
    s_ideal_cm = float(s_start_cm)
    phi_rad = np.deg2rad(phi_start_deg)

    prev_angles_deg = None
    commands: list[dict] = []

    ds_nominal_cm = PITCH_CM_PER_ROT * (DELTA_PHI_RAD / (2.0 * np.pi))

    for k in range(MAX_STEPS):
        t_s = k * DT_S

        if s_pred_cm >= geometry.total_length_cm:
            break

        q = geometry.query(s_pred_cm)
        pos_cm = q["pos_cm"]
        T_hat = q["T_hat"]
        N_hat = q["N_hat"]
        B_hat = q["B_hat"]

        B_des_hat = desired_field_direction(N_hat, B_hat, phi_rad)

        result = solver.solve_step(
            pos_cm=(float(pos_cm[0]), float(pos_cm[1])),
            desired_dir=B_des_hat,
            N_hat=N_hat,
            B_hat=B_hat,
            prev_angles_deg=prev_angles_deg,
        )

        chosen = solver.pick_best_solution(result["solutions"], prev_angles_deg)
        if chosen is None:
            break

        theta_x_deg = chosen["theta_x_deg"]
        theta_y_deg = chosen["theta_y_deg"]
        Bmag_mT = chosen["Bmag_mT"]
        align_cos = chosen["align_cos"]

        speed_scale = solver.speed_scale_from_B(Bmag_mT)
        ds_pred_cm = ds_nominal_cm * speed_scale

        phi_used_rad = result["phi_used_rad"]
        phi_next_rad = phi_used_rad + solver.phase_sign * DELTA_PHI_RAD

        commands.append(
            {
                "k": k,
                "t_s": t_s,
                "s_ideal_cm": s_ideal_cm,
                "s_pred_cm": s_pred_cm,
                "x_pred_cm": float(pos_cm[0]),
                "y_pred_cm": float(pos_cm[1]),
                "phi_cmd_deg": float(np.rad2deg(phi_rad) % 360.0),
                "phi_used_deg": float(np.rad2deg(phi_used_rad) % 360.0),
                "phase_steps": result["phase_steps"],
                "Bdes_x": float(B_des_hat[0]),
                "Bdes_y": float(B_des_hat[1]),
                "Bdes_z": float(B_des_hat[2]),
                "Bused_x": float(result["desired_used_hat"][0]),
                "Bused_y": float(result["desired_used_hat"][1]),
                "Bused_z": float(result["desired_used_hat"][2]),
                "theta_x_deg": float(theta_x_deg),
                "theta_y_deg": float(theta_y_deg),
                "Bmag_mT": float(Bmag_mT),
                "align_cos": float(align_cos),
                "speed_scale": float(speed_scale),
                "ds_nominal_cm": float(ds_nominal_cm),
                "ds_pred_cm": float(ds_pred_cm),
            }
        )

        prev_angles_deg = (theta_x_deg, theta_y_deg)
        s_ideal_cm += ds_nominal_cm
        s_pred_cm += ds_pred_cm
        phi_rad = phi_next_rad

    write_commands_csv(commands, output_csv)
    return commands


if __name__ == "__main__":
    waypoints_cm = [
        (1.0, 7.5),
        (14.0, 7.5),
    ]

    commands = plan_open_loop(
        waypoints_cm=waypoints_cm,
        s_start_cm=0.0,
        phi_start_deg=0.0,
        output_csv=OUTPUT_CSV,
    )

    print("=== plan_open_loop ===")
    print(f"commands_written = {len(commands)}")
    print(f"output_csv       = {Path(OUTPUT_CSV).resolve()}")

    if commands:
        first = commands[0]
        last = commands[-1]

        print(f"first_xy_cm      = ({first['x_pred_cm']:.3f}, {first['y_pred_cm']:.3f})")
        print(f"last_xy_cm       = ({last['x_pred_cm']:.3f}, {last['y_pred_cm']:.3f})")
        print(f"last_s_pred_cm   = {last['s_pred_cm']:.3f}")
        print(f"last_Bmag_mT     = {last['Bmag_mT']:.3f}")
        print(f"last_speed_scale = {last['speed_scale']:.3f}")