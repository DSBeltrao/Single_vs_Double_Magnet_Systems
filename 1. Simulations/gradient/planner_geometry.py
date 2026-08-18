"""
planner_geometry.py

Reference-path geometry for open-loop microrobot planning.

This module stores a 2D path in pool coordinates [cm] and provides
fast local geometry queries by arc length s [cm].

Coordinate convention
---------------------
- Pool origin is the bottom-left corner.
- x increases to the right.
- y increases upward.
- z = 0 is the pool plane.
- Pool size is assumed to be 15 cm x 15 cm unless changed.

Why this module exists
----------------------
The sequential planner should not precompute a full helix in space.
That would become inconsistent once the predicted motion changes.

Instead, this module stores only the fixed reference path and returns
the local path geometry at the current predicted arc length:
- position
- tangent
- planar Bishop frame

The helix direction is then generated later from:
    B_des_hat = cos(phi)*N_hat + sin(phi)*B_hat

For a planar path, the Bishop frame is especially simple and stable:
- T_hat is the path tangent in the xy-plane
- N_hat is T_hat rotated by +90° in-plane
- B_hat is fixed +z

This avoids torsion and frame flips.
"""

from __future__ import annotations

import numpy as np


class PathGeometry:
    """
    Piecewise-linear 2D reference path with arc-length queries.

    Parameters
    ----------
    waypoints_cm : array-like of shape (N, 2) or (N, 3)
        Reference path waypoints in pool coordinates [cm].
        If z is omitted, z=0 is assumed.

    pool_size_cm : float, optional
        Pool side length in cm. Used only for validation.
    """

    def __init__(self, waypoints_cm, pool_size_cm: float = 15.0) -> None:
        self.pool_size_cm = float(pool_size_cm)

        pts = np.asarray(waypoints_cm, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            raise ValueError("waypoints_cm must contain at least 2 points.")

        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        elif pts.shape[1] != 3:
            raise ValueError("waypoints_cm must have shape (N,2) or (N,3).")

        self._validate_points(pts)
        self.points_cm = pts

        # Segment vectors and lengths define the arc-length parameterization.
        seg_vecs = self.points_cm[1:] - self.points_cm[:-1]
        seg_lens = np.linalg.norm(seg_vecs[:, :2], axis=1)

        if np.any(seg_lens < 1e-12):
            raise ValueError("Consecutive waypoints must not be identical.")

        self.seg_vecs_cm = seg_vecs
        self.seg_lens_cm = seg_lens
        self.num_segments = len(seg_lens)

        # Cumulative arc-length table:
        # s_table[i] is the arc length at waypoint i.
        self.s_table_cm = np.zeros(len(self.points_cm))
        self.s_table_cm[1:] = np.cumsum(self.seg_lens_cm)

        self.total_length_cm = float(self.s_table_cm[-1])

        # For a piecewise-linear path, the tangent is constant on each segment.
        self.T_segs = self._compute_segment_tangents()

        # For a planar path, the Bishop frame is the fast in-plane rotation.
        self.N_segs, self.B_segs = self._compute_planar_bishop_frame()

    def _validate_points(self, pts: np.ndarray) -> None:
        """
        Check that waypoints lie inside the pool plane.

        z is allowed but must be zero for the current 2D planner.
        """
        x_ok = np.all((0.0 <= pts[:, 0]) & (pts[:, 0] <= self.pool_size_cm))
        y_ok = np.all((0.0 <= pts[:, 1]) & (pts[:, 1] <= self.pool_size_cm))
        z_ok = np.all(np.abs(pts[:, 2]) < 1e-12)

        if not x_ok or not y_ok:
            raise ValueError("All waypoints must lie inside the pool bounds.")
        if not z_ok:
            raise ValueError("planner_geometry assumes a planar path with z=0.")

    def _compute_segment_tangents(self) -> np.ndarray:
        """
        Return one unit tangent per segment.

        Each tangent is constant over its segment because the path is piecewise linear.
        """
        T = np.zeros_like(self.seg_vecs_cm)
        T[:, :2] = self.seg_vecs_cm[:, :2] / self.seg_lens_cm[:, None]
        return T

    def _compute_planar_bishop_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the planar Bishop frame for each segment.

        For a planar path:
            N_hat = [-Ty, Tx, 0]
            B_hat = [ 0,  0, 1]

        This is the exact rotation-minimizing frame for the current 2D setting.
        """
        N = np.zeros_like(self.T_segs)
        N[:, 0] = -self.T_segs[:, 1]
        N[:, 1] =  self.T_segs[:, 0]

        B = np.zeros_like(self.T_segs)
        B[:, 2] = 1.0
        return N, B

    def clamp_s(self, s_cm: float) -> float:
        """
        Clamp arc length to the valid path interval [0, total_length].
        """
        return float(np.clip(s_cm, 0.0, self.total_length_cm))

    def _segment_index_from_s(self, s_cm: float) -> int:
        """
        Return the segment index containing s.

        If s is exactly at the path end, use the last segment.
        """
        s_cm = self.clamp_s(s_cm)

        if s_cm >= self.total_length_cm:
            return self.num_segments - 1

        # searchsorted gives the first waypoint index whose s_table > s.
        # The containing segment is then one index earlier.
        idx = np.searchsorted(self.s_table_cm, s_cm, side="right") - 1
        return int(np.clip(idx, 0, self.num_segments - 1))

    def query(self, s_cm: float) -> dict:
        """
        Return local geometry at arc length s [cm].

        Parameters
        ----------
        s_cm : float
            Arc length along the reference path [cm].

        Returns
        -------
        dict with keys:
            s_cm       : clamped arc length [cm]
            seg_idx    : segment index
            alpha      : local interpolation fraction in the segment
            pos_cm     : position [x,y,z] in cm
            T_hat      : unit tangent
            N_hat      : planar Bishop normal
            B_hat      : planar Bishop binormal (+z)

        Notes
        -----
        - The position is linearly interpolated within the segment.
        - The frame is piecewise constant because the path is piecewise linear.
        - This is intentional: the sequential planner is the layer that turns
          this local frame into a rotating helical field command.
        """
        s_cm = self.clamp_s(s_cm)
        seg_idx = self._segment_index_from_s(s_cm)

        s0 = self.s_table_cm[seg_idx]
        L = self.seg_lens_cm[seg_idx]
        alpha = (s_cm - s0) / L if L > 0 else 0.0

        p0 = self.points_cm[seg_idx]
        p1 = self.points_cm[seg_idx + 1]
        pos_cm = (1.0 - alpha) * p0 + alpha * p1

        return {
            "s_cm": s_cm,
            "seg_idx": seg_idx,
            "alpha": float(alpha),
            "pos_cm": pos_cm.copy(),
            "T_hat": self.T_segs[seg_idx].copy(),
            "N_hat": self.N_segs[seg_idx].copy(),
            "B_hat": self.B_segs[seg_idx].copy(),
        }

    def nearest_path_sample(self, pos_cm: np.ndarray, num_samples: int = 500) -> dict:
        """
        Return the nearest sampled point on the path to a given position.

        This is not needed for the must-have planner, but it is useful later
        for path-attraction or future feedback.

        Parameters
        ----------
        pos_cm : array-like shape (2,) or (3,)
            Query point in pool coordinates [cm].

        num_samples : int
            Number of arc-length samples used in the search.

        Returns
        -------
        Same dictionary format as query(), but at the nearest sampled arc length.
        """
        pos_cm = np.asarray(pos_cm, dtype=float)
        if pos_cm.shape == (2,):
            pos_cm = np.array([pos_cm[0], pos_cm[1], 0.0])

        s_samples = np.linspace(0.0, self.total_length_cm, num_samples)
        path_samples = np.array([self.query(s)["pos_cm"] for s in s_samples])

        d2 = np.sum((path_samples - pos_cm[None, :])**2, axis=1)
        i = int(np.argmin(d2))
        return self.query(float(s_samples[i]))


if __name__ == "__main__":
    # Simple straight-line test path across the pool center.
    waypoints_cm = [
        (1.0, 7.5),
        (14.0, 7.5),
        (14.0, 14.0),
        (1.0, 14.0),
    ]

    geom = PathGeometry(waypoints_cm)

    print("=== planner_geometry test ===")
    print(f"total_length_cm = {geom.total_length_cm:.3f}")

    # Query a few positions along the path.
    for s in [0.0, 2.0, 5.0, 10.0, 13.0, 16.0, 20.0, 25.0, geom.total_length_cm]:
        q = geom.query(s)
        print(f"\ns = {q['s_cm']:.3f} cm")
        print(f"seg_idx = {q['seg_idx']}")
        print(f"alpha   = {q['alpha']:.3f}")
        print(f"pos_cm  = {q['pos_cm']}")
        print(f"T_hat   = {q['T_hat']}")
        print(f"N_hat   = {q['N_hat']}")
        print(f"B_hat   = {q['B_hat']}")