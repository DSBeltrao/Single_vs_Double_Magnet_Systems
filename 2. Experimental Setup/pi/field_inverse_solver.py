"""
field_inverse_solver.py

Invert a desired magnetic-field direction at a pool position into absolute
angles for two 1-DOF permanent magnets.

This module is stateless:
- it does not update arc length s
- it does not update helix phase phi
- it does not store previous angles

Those belong to the sequential planner.

Coordinate convention
---------------------
- Pool coordinates are in cm, origin at the bottom-left corner.
- The pool lies in z = 0.
- Magnet centers are above and below the pool center along z.
- Magnet 1 rotates about x.
- Magnet 2 rotates about y.

Rotation convention
-------------------
The robot design fixes one valid field-rotation sense for propulsion.
That sense is encoded once in `phase_sign`.
"""

from __future__ import annotations

import numpy as np


class FieldInverseSolver:
    """Dipole/Newton inverse solver for two rotating permanent magnets."""

    MU0_OVER_4PI = 1e-7
    TWO_PI = 2.0 * np.pi

    def __init__(
        self,
        pool_size_cm: float = 15.0,
        magnet_offset_z_cm: float = 20.5,
        dipole_moment_Am2: float = 66.0,
        cone_default_deg: float = 5.0,
        cone_max_deg: float = 12.0,
        dphi_deg: float = 20.0,
        phase_sign: float = -1.0,
        B_expand_mT: float = 6.0,
        B_sync_mT: float = 4.0,
        B_min_mT: float = 0.5,
        local_span_deg: float = 30.0,
        local_step_deg: float = 15.0,
        global_seeds_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
        w_angle: float = 1.0,
        w_mag: float = 2.0,
        w_align: float = 0.05,
    ) -> None:
        self.pool_size_cm = float(pool_size_cm)
        self.magnet_offset_z_cm = float(magnet_offset_z_cm)
        self.dipole_moment_Am2 = float(dipole_moment_Am2)

        self.cone_default_deg = float(cone_default_deg)
        self.cone_max_deg = float(cone_max_deg)

        self.dphi_deg = float(dphi_deg)
        self.max_phase_steps = int(np.ceil(360.0 / self.dphi_deg))

        self.phase_sign = float(np.sign(phase_sign))
        if self.phase_sign == 0.0:
            raise ValueError("phase_sign must be +1 or -1.")

        self.B_expand_mT = float(B_expand_mT)
        self.B_sync_mT = float(B_sync_mT)
        self.B_min_mT = float(B_min_mT)

        self.local_span_deg = float(local_span_deg)
        self.local_step_deg = float(local_step_deg)
        self.global_seeds_deg = tuple(float(v) for v in global_seeds_deg)

        self.w_angle = float(w_angle)
        self.w_mag = float(w_mag)
        self.w_align = float(w_align)

        self.center_m = np.array(
            [self.pool_size_cm / 2.0, self.pool_size_cm / 2.0, 0.0], dtype=float
        ) * 1e-2

        self.p1_m = self.center_m + np.array(
            [0.0, 0.0, +self.magnet_offset_z_cm], dtype=float
        ) * 1e-2

        self.p2_m = self.center_m + np.array(
            [0.0, 0.0, -self.magnet_offset_z_cm], dtype=float
        ) * 1e-2

        self.ex = np.array([1.0, 0.0, 0.0], dtype=float)
        self.ey = np.array([0.0, 1.0, 0.0], dtype=float)
        self.ez = np.array([0.0, 0.0, 1.0], dtype=float)

        self.global_seeds_rad = self._make_global_seeds()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray | None:
        n = np.linalg.norm(v)
        if n < 1e-15:
            return None
        return v / n

    @staticmethod
    def _wrap_deg(x: float) -> float:
        return float(x % 360.0)

    @staticmethod
    def _ang_dist_deg(a: float, b: float) -> float:
        return abs(((a - b + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _solve_2x2(
        A11: float, A12: float, A21: float, A22: float, b1: float, b2: float
    ) -> tuple[float, float] | None:
        det = A11 * A22 - A12 * A21
        if abs(det) < 1e-14:
            return None
        inv = 1.0 / det
        x1 = inv * (A22 * b1 - A12 * b2)
        x2 = inv * (-A21 * b1 + A11 * b2)
        return x1, x2

    @staticmethod
    def _dir_on_plane(N_hat: np.ndarray, B_hat: np.ndarray, phi_rad: float) -> np.ndarray:
        return np.cos(phi_rad) * N_hat + np.sin(phi_rad) * B_hat

    def _perp_basis_from_dhat(
        self, dhat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if abs(dhat[2]) < 0.9:
            tmp = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)

        e1 = np.cross(dhat, tmp)
        e1 = self._normalize(e1)
        if e1 is None:
            return None, None

        e2 = np.cross(dhat, e1)
        return e1, e2

    def _adaptive_cone_deg(self, Bmag_mT: float) -> float:
        if Bmag_mT >= self.B_expand_mT:
            return self.cone_default_deg
        if Bmag_mT <= self.B_min_mT:
            return self.cone_max_deg

        frac = (self.B_expand_mT - Bmag_mT) / (self.B_expand_mT - self.B_min_mT)
        return self.cone_default_deg + frac * (self.cone_max_deg - self.cone_default_deg)

    def _G_matrix(self, rho_m: np.ndarray) -> np.ndarray | None:
        R = np.linalg.norm(rho_m)
        if R < 1e-12:
            return None
        rhat = rho_m / R
        rrT = np.outer(rhat, rhat)
        return self.MU0_OVER_4PI * (3.0 * rrT - np.eye(3)) / (R**3)

    def _build_coeffs_at_point(
        self, pos_cm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        pos_m = np.array([pos_cm[0], pos_cm[1], 0.0], dtype=float) * 1e-2

        G1 = self._G_matrix(pos_m - self.p1_m)
        G2 = self._G_matrix(pos_m - self.p2_m)
        if G1 is None or G2 is None:
            return None

        a = self.dipole_moment_Am2 * (G1 @ (-self.ey))
        b = self.dipole_moment_Am2 * (G1 @ ( self.ez))
        c = self.dipole_moment_Am2 * (G2 @ ( self.ex))
        d = self.dipole_moment_Am2 * (G2 @ ( self.ez))
        return a, b, c, d

    def _newton_solve_direction(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        tx0: float,
        ty0: float,
        iters: int = 15,
        tol: float = 1e-10,
    ) -> tuple[float, float, bool]:
        tx, ty = tx0, ty0

        for _ in range(iters):
            sx, cx = np.sin(tx), np.cos(tx)
            sy, cy = np.sin(ty), np.cos(ty)

            B = a * sx + b * cx + c * sy + d * cy
            r1 = float(e1 @ B)
            r2 = float(e2 @ B)

            if (r1 * r1 + r2 * r2) < (tol * tol):
                return tx, ty, True

            dB_dtx = a * cx - b * sx
            dB_dty = c * cy - d * sy

            J11 = float(e1 @ dB_dtx)
            J12 = float(e1 @ dB_dty)
            J21 = float(e2 @ dB_dtx)
            J22 = float(e2 @ dB_dty)

            step = self._solve_2x2(J11, J12, J21, J22, r1, r2)
            if step is None:
                return tx, ty, False

            tx = (tx - step[0]) % self.TWO_PI
            ty = (ty - step[1]) % self.TWO_PI

        return tx, ty, False

    def _make_global_seeds(self) -> list[tuple[float, float]]:
        seeds = []
        for sx in self.global_seeds_deg:
            for sy in self.global_seeds_deg:
                seeds.append((np.deg2rad(sx), np.deg2rad(sy)))
        return seeds

    def _make_local_seeds(self, prev_angles_deg: tuple[float, float]) -> list[tuple[float, float]]:
        px, py = prev_angles_deg
        offsets = np.arange(
            -self.local_span_deg,
            self.local_span_deg + 1e-9,
            self.local_step_deg,
        )

        seeds = []
        for dx in offsets:
            for dy in offsets:
                tx = np.deg2rad((px + dx) % 360.0)
                ty = np.deg2rad((py + dy) % 360.0)
                seeds.append((tx, ty))
        return seeds

    def _run_seed_list(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        dhat: np.ndarray,
        seed_list: list[tuple[float, float]],
        dedup_tol_deg: float = 1.0,
    ) -> list[dict]:
        e1, e2 = self._perp_basis_from_dhat(dhat)
        if e1 is None:
            return []

        sols: list[dict] = []

        for tx0, ty0 in seed_list:
            tx, ty, ok = self._newton_solve_direction(a, b, c, d, e1, e2, tx0, ty0)
            if not ok:
                continue

            B = a * np.sin(tx) + b * np.cos(tx) + c * np.sin(ty) + d * np.cos(ty)
            Bmag_T = float(np.linalg.norm(B))
            if Bmag_T < 1e-18:
                continue

            Bmag_mT = Bmag_T * 1e3
            if Bmag_mT < self.B_min_mT:
                continue

            Bhat = B / Bmag_T
            align = float(Bhat @ dhat)

            cone_deg = self._adaptive_cone_deg(Bmag_mT)
            cos_thresh = float(np.cos(np.deg2rad(cone_deg)))
            if align < cos_thresh:
                continue

            tx_deg = self._wrap_deg(np.rad2deg(tx))
            ty_deg = self._wrap_deg(np.rad2deg(ty))

            duplicate = False
            for s in sols:
                if (
                    self._ang_dist_deg(tx_deg, s["theta_x_deg"]) < dedup_tol_deg
                    and self._ang_dist_deg(ty_deg, s["theta_y_deg"]) < dedup_tol_deg
                ):
                    duplicate = True
                    break

            if not duplicate:
                sols.append(
                    {
                        "theta_x_deg": tx_deg,
                        "theta_y_deg": ty_deg,
                        "align_cos": align,
                        "Bmag_mT": Bmag_mT,
                        "cone_used_deg": cone_deg,
                        "Bhat": Bhat.copy(),
                    }
                )

        sols.sort(key=lambda s: (-s["Bmag_mT"], -s["align_cos"]))
        return sols

    def find_all_solutions(
        self,
        pos_cm: tuple[float, float],
        desired_dir: np.ndarray,
        prev_angles_deg: tuple[float, float] | None = None,
        dedup_tol_deg: float = 1.0,
    ) -> list[dict]:
        dhat = self._normalize(desired_dir)
        if dhat is None:
            return []

        coeffs = self._build_coeffs_at_point(np.asarray(pos_cm, dtype=float))
        if coeffs is None:
            return []

        a, b, c, d = coeffs
        all_sols: list[dict] = []

        if prev_angles_deg is not None:
            local_sols = self._run_seed_list(
                a, b, c, d, dhat,
                self._make_local_seeds(prev_angles_deg),
                dedup_tol_deg=dedup_tol_deg,
            )
            all_sols.extend(local_sols)

            if local_sols:
                B_best_local = max(s["Bmag_mT"] for s in local_sols)
                if B_best_local >= self.B_expand_mT:
                    return sorted(all_sols, key=lambda s: (-s["Bmag_mT"], -s["align_cos"]))

        global_sols = self._run_seed_list(
            a, b, c, d, dhat,
            self.global_seeds_rad,
            dedup_tol_deg=dedup_tol_deg,
        )

        for gs in global_sols:
            duplicate = False
            for ls in all_sols:
                if (
                    self._ang_dist_deg(gs["theta_x_deg"], ls["theta_x_deg"]) < dedup_tol_deg
                    and self._ang_dist_deg(gs["theta_y_deg"], ls["theta_y_deg"]) < dedup_tol_deg
                ):
                    duplicate = True
                    break
            if not duplicate:
                all_sols.append(gs)

        all_sols.sort(key=lambda s: (-s["Bmag_mT"], -s["align_cos"]))
        return all_sols

    def solve_step(
        self,
        pos_cm: tuple[float, float],
        desired_dir: np.ndarray,
        N_hat: np.ndarray,
        B_hat: np.ndarray,
        prev_angles_deg: tuple[float, float] | None = None,
    ) -> dict:
        """
        Solve one feedforward step using the caller's local frame.

        The planner and the solver must use the same (N_hat, B_hat) basis.
        Rebuilding a new plane basis from the tangent would introduce a 180°
        phase ambiguity.
        """
        dhat = self._normalize(np.asarray(desired_dir, dtype=float))
        if dhat is None:
            return {
                "reachable_direct": False,
                "phi_used_rad": None,
                "phase_steps": None,
                "desired_used_hat": None,
                "solutions": [],
            }

        N_hat = self._normalize(np.asarray(N_hat, dtype=float))
        B_hat = self._normalize(np.asarray(B_hat, dtype=float))
        if N_hat is None or B_hat is None:
            return {
                "reachable_direct": False,
                "phi_used_rad": None,
                "phase_steps": None,
                "desired_used_hat": None,
                "solutions": [],
            }

        x = float(dhat @ N_hat)
        y = float(dhat @ B_hat)
        if abs(x) < 1e-12 and abs(y) < 1e-12:
            return {
                "reachable_direct": False,
                "phi_used_rad": None,
                "phase_steps": None,
                "desired_used_hat": None,
                "solutions": [],
            }

        phi0 = float(np.arctan2(y, x))

        sols0 = self.find_all_solutions(
            pos_cm=pos_cm,
            desired_dir=dhat,
            prev_angles_deg=prev_angles_deg,
        )
        if sols0:
            return {
                "reachable_direct": True,
                "phi_used_rad": phi0,
                "phase_steps": 0,
                "desired_used_hat": dhat.copy(),
                "solutions": sols0,
            }

        dphi = np.deg2rad(self.dphi_deg)
        for k in range(1, self.max_phase_steps + 1):
            phi_try = phi0 + self.phase_sign * k * dphi
            dir_try = self._dir_on_plane(N_hat, B_hat, phi_try)

            sols = self.find_all_solutions(
                pos_cm=pos_cm,
                desired_dir=dir_try,
                prev_angles_deg=prev_angles_deg,
            )
            if sols:
                return {
                    "reachable_direct": False,
                    "phi_used_rad": phi_try,
                    "phase_steps": k,
                    "desired_used_hat": dir_try.copy(),
                    "solutions": sols,
                }

        return {
            "reachable_direct": False,
            "phi_used_rad": None,
            "phase_steps": None,
            "desired_used_hat": None,
            "solutions": [],
        }

    def pick_best_solution(
        self,
        solutions: list[dict],
        prev_angles_deg: tuple[float, float] | None,
    ) -> dict | None:
        if not solutions:
            return None

        def field_penalty(Bmag_mT: float) -> float:
            if Bmag_mT < self.B_min_mT:
                return 1000.0
            return max(0.0, self.B_sync_mT - Bmag_mT) ** 2

        if prev_angles_deg is None:
            best = None
            best_cost = 1e18
            for s in solutions:
                cost = self.w_mag * field_penalty(s["Bmag_mT"]) - self.w_align * s["align_cos"]
                if cost < best_cost:
                    best_cost = cost
                    best = s
            return best

        px, py = prev_angles_deg
        best = None
        best_cost = 1e18

        for s in solutions:
            tx = s["theta_x_deg"]
            ty = s["theta_y_deg"]

            angle_cost = (
                self._ang_dist_deg(tx, px) ** 2
                + self._ang_dist_deg(ty, py) ** 2
            )
            mag_pen = field_penalty(s["Bmag_mT"])

            cost = (
                self.w_angle * angle_cost
                + self.w_mag * mag_pen
                - self.w_align * s["align_cos"]
            )

            if cost < best_cost:
                best_cost = cost
                best = s

        return best

    def speed_scale_from_B(self, Bmag_mT: float) -> float:
        return float(min(1.0, Bmag_mT / self.B_sync_mT))


if __name__ == "__main__":
    solver = FieldInverseSolver(
        dipole_moment_Am2=66.0,
        phase_sign=-1.0,
        B_expand_mT=6.0,
        B_sync_mT=4.0,
        B_min_mT=0.5,
        cone_default_deg=5.0,
        cone_max_deg=12.0,
    )

    pos_cm = (7.5, 7.5)
    desired_dir = np.array([0.0, 1.0, 0.0])
    N_hat = np.array([0.0, 1.0, 0.0])
    B_hat = np.array([0.0, 0.0, 1.0])
    prev_angles = (0.0, 0.0)

    result = solver.solve_step(
        pos_cm=pos_cm,
        desired_dir=desired_dir,
        N_hat=N_hat,
        B_hat=B_hat,
        prev_angles_deg=prev_angles,
    )

    print("=== field_inverse_solver test ===")
    print(f"reachable_direct = {result['reachable_direct']}")
    print(f"phase_steps      = {result['phase_steps']}")
    print(f"num_solutions    = {len(result['solutions'])}")

    chosen = solver.pick_best_solution(result["solutions"], prev_angles)
    if chosen is not None:
        print(f"theta_x_deg      = {chosen['theta_x_deg']:.3f}")
        print(f"theta_y_deg      = {chosen['theta_y_deg']:.3f}")
        print(f"align_cos        = {chosen['align_cos']:.6f}")
        print(f"Bmag_mT          = {chosen['Bmag_mT']:.6f}")
        print(f"cone_used_deg    = {chosen['cone_used_deg']:.3f}")
        print(f"speed_scale      = {solver.speed_scale_from_B(chosen['Bmag_mT']):.6f}")