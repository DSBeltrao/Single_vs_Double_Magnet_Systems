"""
planner_geometry_single_magnet.py

Geometry helper for the sequential open-loop planner.
Same role as the original planner_geometry.py:
- stores a 2D polyline reference path
- returns position and local frame at arc length s

The local frame is:
- T_hat: tangent along the path
- N_hat: left normal in the pool plane
- B_hat: +z
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        return None
    return v / n


class PathGeometry:
    def __init__(self, waypoints_cm):
        pts = np.asarray(waypoints_cm, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("waypoints_cm must be shape (N,2), N>=2.")

        segs = pts[1:] - pts[:-1]
        lens = np.linalg.norm(segs, axis=1)

        if np.any(lens <= 0):
            raise ValueError("Consecutive waypoints must be distinct.")

        self.pts = pts
        self.segs = segs
        self.lens = lens
        self.cum = np.concatenate([[0.0], np.cumsum(lens)])
        self.total_length_cm = float(self.cum[-1])

    def query(self, s_cm: float) -> dict:
        s_cm = float(np.clip(s_cm, 0.0, self.total_length_cm))

        idx = np.searchsorted(self.cum, s_cm, side="right") - 1
        idx = min(idx, len(self.lens) - 1)

        s0 = self.cum[idx]
        ds = s_cm - s0
        u = ds / self.lens[idx]

        p0 = self.pts[idx]
        p1 = self.pts[idx + 1]
        pos_cm = (1.0 - u) * p0 + u * p1

        T_xy = self.segs[idx] / self.lens[idx]
        N_xy = np.array([-T_xy[1], T_xy[0]], dtype=float)

        return {
            "pos_cm": pos_cm,
            "T_hat": np.array([T_xy[0], T_xy[1], 0.0], dtype=float),
            "N_hat": np.array([N_xy[0], N_xy[1], 0.0], dtype=float),
            "B_hat": np.array([0.0, 0.0, 1.0], dtype=float),
        }


if __name__ == "__main__":
    geom = PathGeometry([(7.5, 7.5), (14.0, 14.0)])
    for s in [0.0, 3.0, 6.5, 13.0]:
        q = geom.query(s)
        print(
            f"s = {s:.2f} cm | "
            f"pos = ({q['pos_cm'][0]:.3f}, {q['pos_cm'][1]:.3f}) cm | "
            f"T = {q['T_hat']} | N = {q['N_hat']}"
        )