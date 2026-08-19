"""
plan_open_loop_pi.py

Pi-facing wrapper around your geometry + inverse solver + open-loop planner.

What it does
------------
1. Receives waypoint-based path requests from the socket server.
2. Runs a sequential open-loop planner.
3. Writes:
   - full command CSV
   - motor-only CSV
   - motor-only JSON

Optional:
- stop-and-align at segment corners by inserting fixed-position alignment rows
  before propulsion continues on the next segment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from planner_geometry import PathGeometry
from field_inverse_solver import FieldInverseSolver


def desired_field_direction(N_hat: np.ndarray, B_hat: np.ndarray, phi_rad: float) -> np.ndarray:
    return np.cos(phi_rad) * N_hat + np.sin(phi_rad) * B_hat


def write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def extract_motor_rows(commands: list[dict]) -> list[dict]:
    motor_rows = []
    for row in commands:
        motor_rows.append(
            {
                "k": int(row["k"]),
                "t_s": float(row["t_s"]),
                "theta_x_deg": float(row["theta_x_deg"]),
                "theta_y_deg": float(row["theta_y_deg"]),
            }
        )
    return motor_rows


def solve_align_hold_sequence(
    geometry: PathGeometry,
    solver: FieldInverseSolver,
    seg_idx_next: int,
    pos_cm: np.ndarray,
    t_start_s: float,
    k_start: int,
    prev_angles_deg: tuple[float, float] | None,
    align_steps: int,
) -> tuple[list[dict], tuple[float, float] | None]:
    """
    Insert fixed-position alignment commands before starting the new segment.

    The target alignment vector is the in-plane perpendicular to the new segment
    tangent, i.e. the planar Bishop normal N_hat of that next segment.
    """
    if align_steps <= 0:
        return [], prev_angles_deg

    T_next = geometry.T_segs[seg_idx_next].copy()
    N_next = geometry.N_segs[seg_idx_next].copy()
    B_next = geometry.B_segs[seg_idx_next].copy()

    result = solver.solve_step(
        pos_cm=(float(pos_cm[0]), float(pos_cm[1])),
        desired_dir=N_next,
        N_hat=N_next,
        B_hat=B_next,
        prev_angles_deg=prev_angles_deg,
    )
    chosen = solver.pick_best_solution(result["solutions"], prev_angles_deg)
    if chosen is None:
        return [], prev_angles_deg

    theta_target_x = float(chosen["theta_x_deg"])
    theta_target_y = float(chosen["theta_y_deg"])
    Bmag_mT = float(chosen["Bmag_mT"])
    align_cos = float(chosen["align_cos"])

    rows = []

    if prev_angles_deg is None:
        tx0, ty0 = theta_target_x, theta_target_y
    else:
        tx0, ty0 = prev_angles_deg

    for j, a in enumerate(np.linspace(0.0, 1.0, align_steps, endpoint=True)):
        tx = (1.0 - a) * tx0 + a * theta_target_x
        ty = (1.0 - a) * ty0 + a * theta_target_y

        rows.append(
            {
                "k": k_start + j,
                "t_s": float(t_start_s + j * 0.0),  # overwritten by caller if wanted
                "mode": "align",
                "seg_idx": int(seg_idx_next),
                "s_ideal_cm": None,
                "s_pred_cm": None,
                "x_pred_cm": float(pos_cm[0]),
                "y_pred_cm": float(pos_cm[1]),
                "phi_cmd_deg": None,
                "phi_used_deg": None,
                "phase_steps": int(result["phase_steps"]) if result["phase_steps"] is not None else None,
                "Bdes_x": float(N_next[0]),
                "Bdes_y": float(N_next[1]),
                "Bdes_z": float(N_next[2]),
                "Bused_x": float(result["desired_used_hat"][0]),
                "Bused_y": float(result["desired_used_hat"][1]),
                "Bused_z": float(result["desired_used_hat"][2]),
                "theta_x_deg": float(tx),
                "theta_y_deg": float(ty),
                "Bmag_mT": Bmag_mT,
                "align_cos": align_cos,
                "speed_scale": 0.0,
                "ds_nominal_cm": 0.0,
                "ds_pred_cm": 0.0,
            }
        )

    return rows, (theta_target_x, theta_target_y)


def plan_open_loop_pi(
    waypoints_cm,
    s_start_cm: float = 0.0,
    phi_start_deg: float = 0.0,
    command_rate_hz: float = 20.0,
    rotation_freq_hz: float = 1.0,
    pitch_cm_per_rot: float = 0.124,
    max_time_s: float = 120.0,
    pool_size_cm: float = 15.0,
    align_steps_per_corner: int = 0,
) -> list[dict]:
    geometry = PathGeometry(waypoints_cm, pool_size_cm=pool_size_cm)

    solver = FieldInverseSolver(
        pool_size_cm=pool_size_cm,
        magnet_offset_z_cm=20.5,
        dipole_moment_Am2=66.0,
        cone_default_deg=5.0,
        cone_max_deg=12.0,
        dphi_deg=20.0,
        phase_sign=-1.0,
        B_expand_mT=6.0,
        B_sync_mT=4.0,
        B_min_mT=0.5,
        local_span_deg=30.0,
        local_step_deg=15.0,
        global_seeds_deg=(0.0, 90.0, 180.0, 270.0),
        w_angle=1.0,
        w_mag=2.0,
        w_align=0.05,
    )

    dt_s = 1.0 / float(command_rate_hz)
    delta_phi_rad = 2.0 * np.pi * float(rotation_freq_hz) * dt_s
    max_steps = int(float(max_time_s) * float(command_rate_hz))
    ds_nominal_cm = float(pitch_cm_per_rot) * (delta_phi_rad / (2.0 * np.pi))

    s_pred_cm = float(s_start_cm)
    s_ideal_cm = float(s_start_cm)
    phi_rad = np.deg2rad(phi_start_deg)

    prev_angles_deg = None
    commands: list[dict] = []

    prev_seg_idx = None
    k_out = 0

    for _ in range(max_steps):
        if s_pred_cm >= geometry.total_length_cm:
            break

        q = geometry.query(s_pred_cm)
        pos_cm = q["pos_cm"]
        seg_idx = int(q["seg_idx"])
        N_hat = q["N_hat"]
        B_hat = q["B_hat"]

        # Optional stop-and-align when entering a new segment.
        if (
            align_steps_per_corner > 0
            and prev_seg_idx is not None
            and seg_idx != prev_seg_idx
        ):
            align_rows, prev_angles_deg = solve_align_hold_sequence(
                geometry=geometry,
                solver=solver,
                seg_idx_next=seg_idx,
                pos_cm=pos_cm,
                t_start_s=k_out * dt_s,
                k_start=k_out,
                prev_angles_deg=prev_angles_deg,
                align_steps=align_steps_per_corner,
            )

            for j, row in enumerate(align_rows):
                row["t_s"] = (k_out + j) * dt_s

            commands.extend(align_rows)
            k_out += len(align_rows)

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

        theta_x_deg = float(chosen["theta_x_deg"])
        theta_y_deg = float(chosen["theta_y_deg"])
        Bmag_mT = float(chosen["Bmag_mT"])
        align_cos = float(chosen["align_cos"])

        speed_scale = float(solver.speed_scale_from_B(Bmag_mT))
        ds_pred_cm = ds_nominal_cm * speed_scale

        phi_used_rad = result["phi_used_rad"]
        phi_next_rad = phi_used_rad + solver.phase_sign * delta_phi_rad

        commands.append(
            {
                "k": k_out,
                "t_s": k_out * dt_s,
                "mode": "propel",
                "seg_idx": seg_idx,
                "s_ideal_cm": float(s_ideal_cm),
                "s_pred_cm": float(s_pred_cm),
                "x_pred_cm": float(pos_cm[0]),
                "y_pred_cm": float(pos_cm[1]),
                "phi_cmd_deg": float(np.rad2deg(phi_rad) % 360.0),
                "phi_used_deg": float(np.rad2deg(phi_used_rad) % 360.0),
                "phase_steps": int(result["phase_steps"]) if result["phase_steps"] is not None else None,
                "Bdes_x": float(B_des_hat[0]),
                "Bdes_y": float(B_des_hat[1]),
                "Bdes_z": float(B_des_hat[2]),
                "Bused_x": float(result["desired_used_hat"][0]),
                "Bused_y": float(result["desired_used_hat"][1]),
                "Bused_z": float(result["desired_used_hat"][2]),
                "theta_x_deg": theta_x_deg,
                "theta_y_deg": theta_y_deg,
                "Bmag_mT": Bmag_mT,
                "align_cos": align_cos,
                "speed_scale": speed_scale,
                "ds_nominal_cm": float(ds_nominal_cm),
                "ds_pred_cm": float(ds_pred_cm),
            }
        )

        prev_angles_deg = (theta_x_deg, theta_y_deg)
        prev_seg_idx = seg_idx
        s_ideal_cm += ds_nominal_cm
        s_pred_cm += ds_pred_cm
        phi_rad = phi_next_rad
        k_out += 1

    return commands


def plan_from_request(request: dict) -> dict:
    waypoints_cm = request["waypoints_cm"]

    s_start_cm = float(request.get("s_start_cm", 0.0))
    phi_start_deg = float(request.get("phi_start_deg", 0.0))
    command_rate_hz = float(request.get("command_rate_hz", 20.0))
    rotation_freq_hz = float(request.get("rotation_freq_hz", 1.0))
    pitch_cm_per_rot = float(request.get("pitch_cm_per_rot", 0.124))
    max_time_s = float(request.get("max_time_s", 120.0))
    pool_size_cm = float(request.get("pool_size_cm", 15.0))
    align_steps_per_corner = int(request.get("align_steps_per_corner", 0))

    output_dir = Path(request.get("output_dir", "."))
    base_name = str(request.get("base_name", "latest"))
    write_full_commands = bool(request.get("write_full_commands", True))
    write_motor_only = bool(request.get("write_motor_only", True))

    commands = plan_open_loop_pi(
        waypoints_cm=waypoints_cm,
        s_start_cm=s_start_cm,
        phi_start_deg=phi_start_deg,
        command_rate_hz=command_rate_hz,
        rotation_freq_hz=rotation_freq_hz,
        pitch_cm_per_rot=pitch_cm_per_rot,
        max_time_s=max_time_s,
        pool_size_cm=pool_size_cm,
        align_steps_per_corner=align_steps_per_corner,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    full_csv = output_dir / f"{base_name}_commands_open_loop.csv"
    motor_csv = output_dir / f"{base_name}_motor_angles.csv"
    motor_json = output_dir / f"{base_name}_motor_angles.json"

    if write_full_commands:
        write_csv(commands, full_csv)

    motor_rows = extract_motor_rows(commands)

    if write_motor_only:
        write_csv(motor_rows, motor_csv)
        write_json(motor_rows, motor_json)

    first_angles_deg = None
    last_angles_deg = None
    if motor_rows:
        first_angles_deg = [motor_rows[0]["theta_x_deg"], motor_rows[0]["theta_y_deg"]]
        last_angles_deg = [motor_rows[-1]["theta_x_deg"], motor_rows[-1]["theta_y_deg"]]

    return {
        "num_commands": len(commands),
        "num_motor_rows": len(motor_rows),
        "full_csv": str(full_csv),
        "motor_csv": str(motor_csv),
        "motor_json": str(motor_json),
        "first_angles_deg": first_angles_deg,
        "last_angles_deg": last_angles_deg,
    }


if __name__ == "__main__":
    request = {
        "waypoints_cm": [
            [7.5, 7.5],
            [1, 7.5],
        ],
        "s_start_cm": 0.0,
        "phi_start_deg": 0.0,
        "command_rate_hz": 20.0,
        "rotation_freq_hz": 1.0,
        "pitch_cm_per_rot": 0.124,
        "max_time_s": 120.0,
        "pool_size_cm": 15.0,
        "output_dir": ".",
        "base_name": "latest",
        "write_full_commands": True,
        "write_motor_only": True,
        "align_steps_per_corner": 0,
    }

    result = plan_from_request(request)
    print(json.dumps(result, indent=2))