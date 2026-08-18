"""
field_inverse_solver_single_magnet.py

Single-magnet inverse solver for the synchronous-tilt model.

Scenario:
- one 4x4x4 cm magnet
- center is 12.5 cm below ROI center
- magnetization starts along +z
- the two motor commands (theta_x, theta_y) act synchronously
  and are interpreted as a single rotation vector [tx, ty, 0]

This keeps the same planner-facing workflow as before:
- solve direct desired direction first
- if needed, phase-step on the N/B plane
- use B magnitude thresholds the same way as before
"""

from __future__ import annotations

import numpy as np


class FieldInverseSolver:
    MU0_OVER_4PI = 1e-7

    def __init__(
        self,
        pool_size_cm: float = 15.0,
        magnet_offset_z_cm: float = 12.5,
        dipole_moment_Am2: float = 66.0,
        cone_default_deg: float = 5.0,
        cone_max_deg: float = 12.0,
        dphi_deg: float = 20.0,
        phase_sign: float = -1.0,
        B_expand_mT: float = 6.0,
        B_sync_mT: float = 4.0,
        B_min_mT: float = 1.0,
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

        self.w_angle = float(w_angle)
        self.w_mag = float(w_mag)
        self.w_align = float(w_align)

        self.center_m = np.array(
            [self.pool_size_cm / 2.0, self.pool_size_cm / 2.0, 0.0],
            dtype=float,
        ) * 1e-2

        self.mag_pos_m = self.center_m + np.array(
            [0.0, 0.0, -self.magnet_offset_z_cm],
            dtype=float,
        ) * 1e-2

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray | None:
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-15:
            return None
        return v / n

    @staticmethod
    def _ang_dist_deg(a: float, b: float) -> float:
        return abs(((a - b + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _dir_on_plane(N_hat: np.ndarray, B_hat: np.ndarray, phi_rad: float) -> np.ndarray:
        v = np.cos(phi_rad) * N_hat + np.sin(phi_rad) * B_hat
        return v / np.linalg.norm(v)

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

    def _field_from_mhat(self, pos_cm: tuple[float, float], m_hat: np.ndarray) -> np.ndarray | None:
        pos_m = np.array([pos_cm[0], pos_cm[1], 0.0], dtype=float) * 1e-2
        G = self._G_matrix(pos_m - self.mag_pos_m)
        if G is None:
            return None
        return self.dipole_moment_Am2 * (G @ m_hat)

    @staticmethod
    def angles_to_mhat_sync(theta_x_deg: float, theta_y_deg: float) -> np.ndarray:
        """
        Order-free synchronous tilt model based on a rotation vector [tx, ty, 0].

        Starting from +z, the rotated magnetization is:
            beta = sqrt(tx^2 + ty^2)
            m_hat = [ (ty/beta) sin(beta), -(tx/beta) sin(beta), cos(beta) ]
        """
        tx = np.deg2rad(theta_x_deg)
        ty = np.deg2rad(theta_y_deg)

        beta = np.hypot(tx, ty)
        if beta < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)

        s = np.sin(beta)
        c = np.cos(beta)

        m_hat = np.array([
            (ty / beta) * s,
            -(tx / beta) * s,
            c,
        ], dtype=float)

        return m_hat / np.linalg.norm(m_hat)

    @staticmethod
    def _mhat_to_sync_angles_deg(m_hat: np.ndarray) -> tuple[float, float] | None:
        """
        Inverse of the rotation-vector synchronous tilt model.

        Forward:
            m_hat = [ (ty/beta) sin(beta), -(tx/beta) sin(beta), cos(beta) ]
            beta = sqrt(tx^2 + ty^2)

        Inverse:
            beta = atan2(sqrt(mx^2+my^2), mz)
            tx   = -beta * my / sin(beta)
            ty   =  beta * mx / sin(beta)
        """
        m_hat = np.asarray(m_hat, dtype=float)
        n = np.linalg.norm(m_hat)
        if n < 1e-15:
            return None

        m_hat = m_hat / n
        mx, my, mz = m_hat
        mz = np.clip(mz, -1.0, 1.0)

        rho = np.hypot(mx, my)
        beta = np.arctan2(rho, mz)
        s = np.sin(beta)

        if s < 1e-12:
            if mz > 0.0:
                return 0.0, 0.0
            return 180.0, 0.0

        tx = -beta * my / s
        ty =  beta * mx / s

        return float(np.degrees(tx)), float(np.degrees(ty))

    def _direct_inverse(self, pos_cm: tuple[float, float], desired_dir: np.ndarray) -> dict | None:
        dhat = self._normalize(desired_dir)
        if dhat is None:
            return None

        pos_m = np.array([pos_cm[0], pos_cm[1], 0.0], dtype=float) * 1e-2
        G = self._G_matrix(pos_m - self.mag_pos_m)
        if G is None:
            return None

        try:
            m_raw = np.linalg.solve(G, dhat)
        except np.linalg.LinAlgError:
            return None

        m_hat = self._normalize(m_raw)
        if m_hat is None:
            return None

        angs = self._mhat_to_sync_angles_deg(m_hat)
        if angs is None:
            return None

        theta_x_deg, theta_y_deg = angs

        B = self._field_from_mhat(pos_cm, m_hat)
        if B is None:
            return None

        Bmag_T = float(np.linalg.norm(B))
        if Bmag_T < 1e-18:
            return None

        Bhat = B / Bmag_T
        align_cos = float(np.dot(Bhat, dhat))
        Bmag_mT = 1e3 * Bmag_T

        if Bmag_mT < self.B_min_mT:
            return None

        cone_deg = self._adaptive_cone_deg(Bmag_mT)
        if align_cos < float(np.cos(np.deg2rad(cone_deg))):
            return None

        return {
            "theta_x_deg": float(theta_x_deg),
            "theta_y_deg": float(theta_y_deg),
            "align_cos": float(align_cos),
            "Bmag_mT": float(Bmag_mT),
            "cone_used_deg": float(cone_deg),
            "Bhat": Bhat.copy(),
            "m_hat": m_hat.copy(),
            "B_vec_T": B.copy(),
        }

    def find_all_solutions(
        self,
        pos_cm: tuple[float, float],
        desired_dir: np.ndarray,
        prev_angles_deg: tuple[float, float] | None = None,
        dedup_tol_deg: float = 1.0,
    ) -> list[dict]:
        sol = self._direct_inverse(pos_cm=pos_cm, desired_dir=desired_dir)
        return [] if sol is None else [sol]

    def solve_step(
        self,
        pos_cm: tuple[float, float],
        desired_dir: np.ndarray,
        N_hat: np.ndarray,
        B_hat: np.ndarray,
        prev_angles_deg: tuple[float, float] | None = None,
    ) -> dict:
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

            angle_cost = self._ang_dist_deg(tx, px) ** 2 + self._ang_dist_deg(ty, py) ** 2
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
    solver = FieldInverseSolver()

    pos_cm = (7.5, 7.5)
    desired_dir = np.array([0.0, 0.4, -0.916515], dtype=float)
    desired_dir /= np.linalg.norm(desired_dir)

    N_hat = np.array([0.0, 1.0, 0.0], dtype=float)
    B_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    result = solver.solve_step(
        pos_cm=pos_cm,
        desired_dir=desired_dir,
        N_hat=N_hat,
        B_hat=B_hat,
        prev_angles_deg=(0.0, 0.0),
    )

    print("reachable_direct:", result["reachable_direct"])
    print("phase_steps:", result["phase_steps"])
    print("n_solutions:", len(result["solutions"]))

    chosen = solver.pick_best_solution(result["solutions"], (0.0, 0.0))
    if chosen is not None:
        print("theta_x_deg:", chosen["theta_x_deg"])
        print("theta_y_deg:", chosen["theta_y_deg"])
        print("Bmag_mT:", chosen["Bmag_mT"])
        print("align_cos:", chosen["align_cos"])
        print("m_hat:", chosen["m_hat"])