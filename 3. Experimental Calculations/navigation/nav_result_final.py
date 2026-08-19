#!/usr/bin/env python3
"""
nav_manual_samples_clean.py

Clean workflow for final navigation analysis.

This script does NOT use automatic robot tracking.

It creates two things at the same time:

1) Thesis figure:
   - manually selected ghost patches are pasted onto the first warped frame
   - desired path is drawn
   - measured path is drawn through the manually clicked robot centers

2) Quantitative data:
   - every ghost sample has the real selected frame number
   - every ghost sample has a manually clicked robot center
   - the script converts these centers to mm
   - velocities are computed between consecutive manually selected samples

Install:
    pip install opencv-python numpy pandas

Run:
    python nav_manual_samples_clean.py --video c-5.5.mp4 --out results

Controls in manual sample mode:
    a / d    previous / next frame
    j / l    jump backward / forward 10 frames
    g        add sample: click robot center, then select ghost ROI
    u        undo last sample
    p        preview ghost image
    Enter    finish samples and save
    Esc      cancel

Coordinate convention after orientation:
    origin = center of the selected 1 cm square
    +x = right
    +y = up
    fixed orientation = rotate warped video 90 degrees clockwise
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Basic display helpers
# ============================================================

def fit_display_size(width, height, max_display=1000):
    scale = min(max_display / max(width, height), 1.0)
    return int(round(width * scale)), int(round(height * scale)), scale


def show_resized(win, img, max_display=1000):
    h, w = img.shape[:2]
    dw, dh, _ = fit_display_size(w, h, max_display)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.imshow(win, cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA))


def draw_text(img, lines, x=10, y=25):
    out = img.copy()
    for line in lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        y += 23
    return out


def select_points(image, title, n_points, instructions, max_display=1000, connect=False):
    h, w = image.shape[:2]
    dw, dh, scale = fit_display_size(w, h, max_display)
    points = []

    def make_display():
        base = draw_text(image.copy(), instructions)
        for i, p in enumerate(points):
            pi = tuple(np.round(p).astype(int))
            cv2.circle(base, pi, 6, (0, 255, 255), -1)
            cv2.putText(base, str(i + 1), (pi[0] + 8, pi[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if connect and len(points) > 1:
            cv2.polylines(base, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
        return cv2.resize(base, (dw, dh), interpolation=cv2.INTER_AREA)

    shown = make_display()

    def mouse(event, x, y, flags, param):
        nonlocal shown
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append(np.array([x / scale, y / scale], dtype=np.float32))
            shown = make_display()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, dw, dh)
    cv2.setMouseCallback(title, mouse)

    while True:
        cv2.imshow(title, shown)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(points) == n_points:
            break
        if key == ord("r"):
            points.clear()
            shown = make_display()
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

    cv2.destroyWindow(title)
    return np.array(points, dtype=np.float32)


def select_roi_scaled(image, title, instructions, max_display=1000):
    h, w = image.shape[:2]
    dw, dh, scale = fit_display_size(w, h, max_display)
    display = draw_text(image.copy(), instructions)
    display = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, dw, dh)
    roi = cv2.selectROI(title, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)

    x, y, rw, rh = roi
    if rw == 0 or rh == 0:
        raise RuntimeError("ROI selection was cancelled or empty.")

    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(rw / scale)),
        int(round(rh / scale)),
    )


# ============================================================
# Geometry and coordinates
# ============================================================

def order_quad_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def center_origin_and_scale_from_four_points(pts):
    ordered = order_quad_points(pts)
    tl, tr, br, bl = ordered

    side_lengths = [
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - tr),
        np.linalg.norm(br - bl),
        np.linalg.norm(bl - tl),
    ]

    origin_px = np.mean(ordered, axis=0).astype(np.float32)

    # selected square is 1 cm = 10 mm
    px_per_mm = float(np.mean(side_lengths)) / 10.0

    return origin_px, px_per_mm, ordered, side_lengths


def apply_orientation_frame(img, mode):
    if mode == "identity":
        return img
    if mode == "flip_h":
        return cv2.flip(img, 1)
    if mode == "flip_v":
        return cv2.flip(img, 0)
    if mode == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    if mode == "rot90_cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if mode == "rot90_ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "transpose":
        return cv2.transpose(img)
    if mode == "anti_transpose":
        return cv2.flip(cv2.transpose(img), -1)
    raise ValueError(f"Unknown orientation mode: {mode}")


def transform_point(p, w, h, mode):
    x, y = float(p[0]), float(p[1])

    if mode == "identity":
        return np.array([x, y], dtype=np.float32)
    if mode == "flip_h":
        return np.array([w - 1 - x, y], dtype=np.float32)
    if mode == "flip_v":
        return np.array([x, h - 1 - y], dtype=np.float32)
    if mode == "rot180":
        return np.array([w - 1 - x, h - 1 - y], dtype=np.float32)
    if mode == "rot90_cw":
        return np.array([h - 1 - y, x], dtype=np.float32)
    if mode == "rot90_ccw":
        return np.array([y, w - 1 - x], dtype=np.float32)
    if mode == "transpose":
        return np.array([y, x], dtype=np.float32)
    if mode == "anti_transpose":
        return np.array([h - 1 - y, w - 1 - x], dtype=np.float32)

    raise ValueError(f"Unknown orientation mode: {mode}")


def choose_orientation_for_negative_quadrant(origin_px, minus_point_px, w, h):
    modes = [
        "identity", "flip_h", "flip_v", "rot180",
        "rot90_cw", "rot90_ccw", "transpose", "anti_transpose"
    ]

    # In image coordinates, bottom-left direction means dx < 0 and dy > 0.
    target = np.array([-1.0, 1.0], dtype=float)
    target /= np.linalg.norm(target)

    best_mode = "identity"
    best_score = -1e9

    for mode in modes:
        o = transform_point(origin_px, w, h, mode)
        m = transform_point(minus_point_px, w, h, mode)
        v = m - o
        n = np.linalg.norm(v)
        if n < 1e-9:
            continue

        score = float(np.dot(v / n, target))
        if score > best_score:
            best_score = score
            best_mode = mode

    return best_mode, best_score


def mm_to_px_xy(x_mm, y_mm, origin_px, px_per_mm):
    ox, oy = origin_px
    return np.array([ox + x_mm * px_per_mm, oy - y_mm * px_per_mm], dtype=np.float32)


def px_to_mm_xy(x_px, y_px, origin_px, px_per_mm):
    ox, oy = origin_px
    return float((x_px - ox) / px_per_mm), float((oy - y_px) / px_per_mm)


# ============================================================
# Floating corner tracking
# ============================================================

def track_corners_all_frames(video_path, initial_corners, preview=False):
    cap = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Could not read first video frame.")

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    p0 = np.asarray(initial_corners, dtype=np.float32).reshape(-1, 1, 2)
    last_good = np.asarray(initial_corners, dtype=np.float32)

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001)
    )

    tracks = [{"frame": 0, "ok": True, "corners": last_good.copy()}]
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

        good = False
        if p1 is not None and st is not None and int(st.sum()) == 4:
            candidate = p1.reshape(-1, 2).astype(np.float32)
            movement = np.linalg.norm(candidate - last_good, axis=1)
            if np.max(movement) < 80:
                last_good = candidate.copy()
                good = True

        p0 = last_good.reshape(-1, 1, 2)
        tracks.append({"frame": frame_idx, "ok": good, "corners": last_good.copy()})

        if preview:
            shown = frame.copy()
            pts = last_good.astype(int)
            cv2.polylines(shown, [pts], True, (0, 255, 255), 2)
            for i, p in enumerate(pts):
                cv2.circle(shown, tuple(p), 6, (0, 255, 255), -1)
                cv2.putText(shown, str(i + 1), (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            show_resized("corner tracking preview", shown)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()
    return tracks


# ============================================================
# Frame loading and warping
# ============================================================

def get_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def warp_oriented_frame(frame, corners_now, dst, out_size, orientation_mode):
    H = cv2.getPerspectiveTransform(corners_now.astype(np.float32), dst)
    warped_raw = cv2.warpPerspective(frame, H, out_size)
    return apply_orientation_frame(warped_raw, orientation_mode)



# ============================================================
# Rotation-start frame selection
# ============================================================

def select_rotation_start_frame(video_path, corner_tracks, dst, out_size, orientation_mode, max_display=1000):
    """
    Browse the warped/oriented video and select the frame where rotation starts.
    This selected frame becomes t = 0 for the quantitative CSV.
    """
    total_frames = len(corner_tracks)
    idx = 0

    print("\nSelect rotation-start frame:")
    print("a/d = previous/next frame")
    print("j/l = jump backward/forward 10 frames")
    print("k/; = jump backward/forward 100 frames")
    print("enter = select current frame as rotation start")

    while True:
        raw = get_frame(video_path, idx)
        if raw is None:
            break

        frame = warp_oriented_frame(
            raw,
            corner_tracks[idx]["corners"],
            dst,
            out_size,
            orientation_mode
        )

        shown = draw_text(frame.copy(), [
            "Select ROTATION START frame",
            "a/d prev/next, j/l +/-10, k/; +/-100",
            "ENTER = use current frame as t = 0"
        ])

        cv2.putText(
            shown,
            f"rotation start candidate: frame {idx}/{total_frames - 1}",
            (10, shown.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        show_resized("select rotation start", shown, max_display=max_display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("a"):
            idx = max(0, idx - 1)
        elif key == ord("d"):
            idx = min(total_frames - 1, idx + 1)
        elif key == ord("j"):
            idx = max(0, idx - 10)
        elif key == ord("l"):
            idx = min(total_frames - 1, idx + 10)
        elif key == ord("k"):
            idx = max(0, idx - 100)
        elif key == ord(";"):
            idx = min(total_frames - 1, idx + 100)
        elif key in (13, 32):
            cv2.destroyWindow("select rotation start")
            print(f"Selected rotation start frame: {idx}")
            return int(idx)
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

    cv2.destroyAllWindows()
    return 0


# ============================================================
# Ghost patch generation
# ============================================================

def paste_patch_with_alpha(canvas, frame, roi, alpha=0.45, feather=10):
    x, y, w, h = [int(v) for v in roi]
    H, W = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 <= x1 or y2 <= y1:
        return canvas

    patch = frame[y1:y2, x1:x2].copy()

    ph, pw = patch.shape[:2]
    mask = np.ones((ph, pw), dtype=np.float32)

    feather = min(feather, ph // 2, pw // 2)
    if feather > 0:
        ramp = np.linspace(0, 1, feather)
        mask[:feather, :] *= ramp[:, None]
        mask[-feather:, :] *= ramp[::-1, None]
        mask[:, :feather] *= ramp[None, :]
        mask[:, -feather:] *= ramp[::-1][None, :]

    mask = (mask * alpha)[..., None]
    roi_canvas = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = roi_canvas * (1 - mask) + patch.astype(np.float32) * mask
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return canvas


def manual_sample_builder(video_path, corner_tracks, dst, out_size, orientation_mode,
                          first_oriented, ghost_alpha, max_display=1000):
    total_frames = len(corner_tracks)
    idx = 0
    ghost = first_oriented.copy()
    samples = []

    print("\nManual ghost/data mode:")
    print("a/d = previous/next frame")
    print("j/l = jump -/+10 frames")
    print("g   = add sample: click robot center, then draw ghost patch ROI")
    print("u   = undo last sample")
    print("p   = preview ghost image")
    print("enter = finish and save")
    print("\nNo fixed frame spacing is assumed. The actual selected frame is saved.")

    def rebuild_ghost():
        rebuilt = first_oriented.copy()
        for row in samples:
            raw_i = get_frame(video_path, int(row["frame"]))
            if raw_i is None:
                continue
            frame_i = warp_oriented_frame(
                raw_i,
                corner_tracks[int(row["frame"])]["corners"],
                dst,
                out_size,
                orientation_mode
            )
            roi_i = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_w"]), int(row["roi_h"]))
            rebuilt = paste_patch_with_alpha(rebuilt, frame_i, roi_i, alpha=ghost_alpha)
        return rebuilt

    while True:
        raw = get_frame(video_path, idx)
        if raw is None:
            break

        frame = warp_oriented_frame(
            raw,
            corner_tracks[idx]["corners"],
            dst,
            out_size,
            orientation_mode
        )

        shown = frame.copy()
        shown = draw_text(shown, [
            "Manual ghost/data mode",
            "a/d prev/next, j/l jump, g add sample, u undo, p preview, ENTER finish",
            "Each sample: click robot center for DATA, then draw ROI for FIGURE"
        ])
        cv2.putText(shown, f"frame {idx}/{total_frames - 1}", (10, shown.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        show_resized("manual sample browser", shown, max_display=max_display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("a"):
            idx = max(0, idx - 1)
        elif key == ord("d"):
            idx = min(total_frames - 1, idx + 1)
        elif key == ord("j"):
            idx = max(0, idx - 10)
        elif key == ord("l"):
            idx = min(total_frames - 1, idx + 10)
        elif key == ord("p"):
            show_resized("current ghost preview", ghost, max_display=max_display)
            cv2.waitKey(0)
            cv2.destroyWindow("current ghost preview")
        elif key == ord("u"):
            if samples:
                removed = samples.pop()
                ghost = rebuild_ghost()
                print(f"Removed sample at frame {removed['frame']}")
            else:
                print("No sample to undo.")
        elif key == ord("g"):
            center = select_points(frame, "click robot center", 1, [
                "Click the physical robot center.",
                "This clicked center is used for the DATA CSV.",
                "Press ENTER/SPACE when done."
            ], max_display=max_display)[0]

            roi = select_roi_scaled(frame, "select ghost patch ROI", [
                "Draw a box around the robot for the FIGURE ghost patch.",
                "This ROI is visual only. Data uses the clicked center.",
                "Press ENTER/SPACE in ROI selector when done."
            ], max_display=max_display)

            ghost = paste_patch_with_alpha(ghost, frame, roi, alpha=ghost_alpha)

            x, y, w, h = roi
            samples.append({
                "sample_index": len(samples),
                "frame": int(idx),
                "data_x_px": float(center[0]),
                "data_y_px": float(center[1]),
                "roi_x": int(x),
                "roi_y": int(y),
                "roi_w": int(w),
                "roi_h": int(h),
                "patch_center_x_px": float(x + w / 2),
                "patch_center_y_px": float(y + h / 2),
            })

            print(
                f"Added sample {len(samples)-1}: "
                f"frame={idx}, center=({center[0]:.1f}, {center[1]:.1f}), roi={roi}"
            )

        elif key in (13, 32):
            break
        elif key == 27:
            raise SystemExit("Cancelled.")

    cv2.destroyAllWindows()
    return ghost, pd.DataFrame(samples)


# ============================================================
# Data conversion and velocity
# ============================================================

def samples_to_dataframes(samples_df, fps, origin_px, px_per_mm, desired_start_mm, desired_end_mm, rotation_start_frame=0):
    if samples_df is None or samples_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    point_df = samples_df.copy().sort_values("frame").reset_index(drop=True)
    point_df["sample_index"] = np.arange(len(point_df))
    point_df["time_video_s"] = point_df["frame"] / float(fps)
    point_df["rotation_start_frame"] = int(rotation_start_frame)
    point_df["time_s"] = (point_df["frame"] - int(rotation_start_frame)) / float(fps)
    point_df["time_from_rotation_start_s"] = point_df["time_s"]

    x_mm = []
    y_mm = []
    for _, row in point_df.iterrows():
        xm, ym = px_to_mm_xy(row["data_x_px"], row["data_y_px"], origin_px, px_per_mm)
        x_mm.append(xm)
        y_mm.append(ym)

    point_df["x_mm"] = x_mm
    point_df["y_mm"] = y_mm

    x0, y0 = desired_start_mm
    x1, y1 = desired_end_mm

    desired_vec = np.array([x1 - x0, y1 - y0], dtype=float)
    desired_len = np.linalg.norm(desired_vec)
    if desired_len < 1e-9:
        raise RuntimeError("Desired start and end are identical.")

    e_parallel = desired_vec / desired_len
    e_perp = np.array([e_parallel[1], -e_parallel[0]])

    dx0 = point_df["x_mm"].to_numpy() - x0
    dy0 = point_df["y_mm"].to_numpy() - y0

    point_df["s_parallel_mm"] = dx0 * e_parallel[0] + dy0 * e_parallel[1]
    point_df["s_perp_mm"] = dx0 * e_perp[0] + dy0 * e_perp[1]

    for col in ["vx_mm_s", "vy_mm_s", "v_parallel_mm_s", "v_perp_mm_s", "speed_mm_s"]:
        point_df[col] = np.nan

    if len(point_df) >= 2:
        t = point_df["time_s"].to_numpy(dtype=float)
        x = point_df["x_mm"].to_numpy(dtype=float)
        y = point_df["y_mm"].to_numpy(dtype=float)

        if len(point_df) >= 3:
            vx = np.gradient(x, t)
            vy = np.gradient(y, t)
        else:
            vx_val = (x[1] - x[0]) / (t[1] - t[0])
            vy_val = (y[1] - y[0]) / (t[1] - t[0])
            vx = np.array([vx_val, vx_val])
            vy = np.array([vy_val, vy_val])

        point_df["vx_mm_s"] = vx
        point_df["vy_mm_s"] = vy
        point_df["v_parallel_mm_s"] = vx * e_parallel[0] + vy * e_parallel[1]
        point_df["v_perp_mm_s"] = vx * e_perp[0] + vy * e_perp[1]
        point_df["speed_mm_s"] = np.sqrt(vx**2 + vy**2)

    segment_rows = []
    for i in range(len(point_df) - 1):
        r0 = point_df.iloc[i]
        r1 = point_df.iloc[i + 1]

        dt = float(r1["time_s"] - r0["time_s"])
        if dt <= 0:
            continue

        dx = float(r1["x_mm"] - r0["x_mm"])
        dy = float(r1["y_mm"] - r0["y_mm"])
        vx = dx / dt
        vy = dy / dt
        v_parallel = vx * e_parallel[0] + vy * e_parallel[1]
        v_perp = vx * e_perp[0] + vy * e_perp[1]

        segment_rows.append({
            "segment_index": i,
            "sample_start": int(r0["sample_index"]),
            "sample_end": int(r1["sample_index"]),
            "frame_start": int(r0["frame"]),
            "frame_end": int(r1["frame"]),
            "time_start_s": float(r0["time_s"]),
            "time_end_s": float(r1["time_s"]),
            "time_mid_s": 0.5 * (float(r0["time_s"]) + float(r1["time_s"])),
            "dt_s": dt,
            "x_start_mm": float(r0["x_mm"]),
            "y_start_mm": float(r0["y_mm"]),
            "x_end_mm": float(r1["x_mm"]),
            "y_end_mm": float(r1["y_mm"]),
            "x_mid_mm": 0.5 * (float(r0["x_mm"]) + float(r1["x_mm"])),
            "y_mid_mm": 0.5 * (float(r0["y_mm"]) + float(r1["y_mm"])),
            "dx_mm": dx,
            "dy_mm": dy,
            "vx_mm_s": vx,
            "vy_mm_s": vy,
            "v_parallel_mm_s": v_parallel,
            "v_perp_mm_s": v_perp,
            "speed_mm_s": float(np.sqrt(vx**2 + vy**2)),
            "s_parallel_mid_mm": 0.5 * (float(r0["s_parallel_mm"]) + float(r1["s_parallel_mm"])),
            "s_perp_mid_mm": 0.5 * (float(r0["s_perp_mm"]) + float(r1["s_perp_mm"])),
        })

    segment_df = pd.DataFrame(segment_rows)
    return point_df, segment_df


# ============================================================
# Final drawing
# ============================================================

def draw_final_overlay(base, origin_px, desired_start_px, desired_end_px,
                       measured_path_px, center_square_px=None):
    out = base.copy()

    # Desired path: yellow
    cv2.line(out, tuple(desired_start_px.astype(int)), tuple(desired_end_px.astype(int)), (0, 255, 255), 3)
    cv2.circle(out, tuple(desired_start_px.astype(int)), 7, (0, 255, 255), -1)
    cv2.circle(out, tuple(desired_end_px.astype(int)), 7, (0, 0, 255), -1)

    # Measured path through clicked centers: cyan.
    # Also draw the segment from origin to the first clicked ghost sample.
    if measured_path_px is not None and len(measured_path_px) > 0:
        first_p = measured_path_px[0].astype(int)
        cv2.line(out, tuple(origin_px.astype(int)), tuple(first_p), (255, 255, 0), 3)

    if measured_path_px is not None and len(measured_path_px) > 1:
        cv2.polylines(out, [measured_path_px.astype(np.int32)], False, (255, 255, 0), 3)
        for p in measured_path_px:
            cv2.circle(out, tuple(p.astype(int)), 4, (255, 255, 0), -1)

    # Origin and axes
    cv2.drawMarker(out, tuple(origin_px.astype(int)), (255, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    # Put label away from the center calibration square.
    cv2.putText(out, "origin (0,0)", tuple((origin_px + np.array([70, -35])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.line(out, tuple(origin_px.astype(int)), tuple((origin_px + np.array([65, -28])).astype(int)),
             (255, 0, 255), 2)

    axis_len = 40
    cv2.arrowedLine(out, tuple(origin_px.astype(int)),
                    tuple((origin_px + np.array([axis_len, 0])).astype(int)),
                    (255, 0, 255), 2)
    cv2.putText(out, "+x", tuple((origin_px + np.array([axis_len + 5, 5])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    cv2.arrowedLine(out, tuple(origin_px.astype(int)),
                    tuple((origin_px + np.array([0, -axis_len])).astype(int)),
                    (255, 0, 255), 2)
    cv2.putText(out, "+y", tuple((origin_px + np.array([5, -axis_len - 5])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    if center_square_px is not None:
        cv2.polylines(out, [center_square_px.astype(np.int32)], True, (255, 0, 255), 2)

    #out = draw_text(out, [
    #    "yellow = desired path",
    #    "cyan = clicked robot-center path",
    #    "transparent patches = ghost samples"
    #])

    return out


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="manual_nav_results_clean")

    ap.add_argument("--pool-width-cm", type=float, default=15.0)
    ap.add_argument("--pool-height-cm", type=float, default=15.0)
    ap.add_argument("--display-px-per-cm", type=float, default=80.0)

    ap.add_argument("--desired-start-mm", nargs=2, type=float, default=None)
    ap.add_argument("--desired-end-mm", nargs=2, type=float, default=None)

    ap.add_argument("--ghost-alpha", type=float, default=0.45)
    ap.add_argument("--no-floating-corners", action="store_true")
    ap.add_argument("--show-corner-preview", action="store_true")

    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()

    if not ok:
        raise SystemExit("Could not read first frame.")

    out_w = int(round(args.pool_width_cm * args.display_px_per_cm))
    out_h = int(round(args.pool_height_cm * args.display_px_per_cm))
    out_size = (out_w, out_h)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32
    )

    # 1) Outer pool/workspace corners
    outer_clicked = select_points(first, "1 - Select outer pool/workspace corners", 4, [
        "Click 4 outer pool/workspace corners.",
        "Any order is okay.",
        "These corners define the perspective warp.",
        "Press ENTER/SPACE when done."
    ])
    outer_ordered = order_quad_points(outer_clicked)

    if args.no_floating_corners:
        corner_tracks = [
            {"frame": i, "ok": True, "corners": outer_ordered.copy()}
            for i in range(total_frames)
        ]
    else:
        corner_tracks = track_corners_all_frames(
            video_path,
            outer_ordered,
            preview=args.show_corner_preview
        )

    # 2) First warped frame
    H_first = cv2.getPerspectiveTransform(corner_tracks[0]["corners"].astype(np.float32), dst)
    first_warped_raw = cv2.warpPerspective(first, H_first, out_size)

    # 3) Center square calibration
    center_square_raw = select_points(first_warped_raw, "2 - Select center 1 cm square", 4, [
        "Click the 4 corners of the 1 cm x 1 cm square around the pool center.",
        "Origin = average of these 4 points.",
        "Scale = average side length = 10 mm.",
        "Press ENTER/SPACE when done."
    ])
    origin_raw, px_per_mm_raw, center_square_ordered_raw, side_lengths = center_origin_and_scale_from_four_points(center_square_raw)

    # 4) Fixed orientation.
    # All videos use the same setup, so no manual (-x,-y) selection is needed.
    # The correct orientation is a 90 degree clockwise rotation of the warped video.
    orientation_mode = "rot90_cw"
    orientation_score = np.nan

    first_oriented = apply_orientation_frame(first_warped_raw, orientation_mode)
    origin_px = transform_point(origin_raw, out_w, out_h, orientation_mode)
    center_square_px = np.array(
        [transform_point(p, out_w, out_h, orientation_mode) for p in center_square_ordered_raw],
        dtype=np.float32
    )
    px_per_mm = px_per_mm_raw

    # 5) Select the frame where rotation actually starts.
    # This makes time_s = 0 correspond to actuation start, not video start.
    rotation_start_frame = select_rotation_start_frame(
        video_path=video_path,
        corner_tracks=corner_tracks,
        dst=dst,
        out_size=out_size,
        orientation_mode=orientation_mode
    )

    # 6) Desired path
    if args.desired_start_mm is None:
        x0 = float(input("Desired path START x [mm] relative to center origin: "))
        y0 = float(input("Desired path START y [mm] relative to center origin: "))
    else:
        x0, y0 = args.desired_start_mm

    if args.desired_end_mm is None:
        x1 = float(input("Desired path END x [mm] relative to center origin: "))
        y1 = float(input("Desired path END y [mm] relative to center origin: "))
    else:
        x1, y1 = args.desired_end_mm

    desired_start_px = mm_to_px_xy(x0, y0, origin_px, px_per_mm)
    desired_end_px = mm_to_px_xy(x1, y1, origin_px, px_per_mm)

    # 7) Manual samples
    ghost_img, samples_raw_df = manual_sample_builder(
        video_path=video_path,
        corner_tracks=corner_tracks,
        dst=dst,
        out_size=out_size,
        orientation_mode=orientation_mode,
        first_oriented=first_oriented,
        ghost_alpha=args.ghost_alpha
    )

    # 8) Convert samples to data
    point_df, segment_df = samples_to_dataframes(
        samples_df=samples_raw_df,
        fps=fps,
        origin_px=origin_px,
        px_per_mm=px_per_mm,
        desired_start_mm=(x0, y0),
        desired_end_mm=(x1, y1),
        rotation_start_frame=rotation_start_frame
    )

    measured_path_px = None
    if not samples_raw_df.empty:
        measured_path_px = samples_raw_df[["data_x_px", "data_y_px"]].to_numpy(dtype=np.float32)

    final_img = draw_final_overlay(
        ghost_img,
        origin_px=origin_px,
        desired_start_px=desired_start_px,
        desired_end_px=desired_end_px,
        measured_path_px=measured_path_px,
        center_square_px=center_square_px
    )

    # 9) Save
    cv2.imwrite(str(out_dir / "FIGURE_manual_ghost_clean.png"), ghost_img)
    cv2.imwrite(str(out_dir / "FIGURE_final_overlay.png"), final_img)
    cv2.imwrite(str(out_dir / "FIGURE_warped_oriented_first_frame.png"), first_oriented)

    samples_raw_df.to_csv(out_dir / "DATA_raw_manual_samples_px.csv", index=False)
    point_df.to_csv(out_dir / "DATA_clicked_centers_positions_velocity.csv", index=False)
    segment_df.to_csv(out_dir / "DATA_clicked_segment_velocity.csv", index=False)

    settings = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "pool_width_cm": args.pool_width_cm,
        "pool_height_cm": args.pool_height_cm,
        "display_px_per_cm": args.display_px_per_cm,
        "px_per_mm": px_per_mm,
        "mm_per_px": 1.0 / px_per_mm,
        "origin_px_x": float(origin_px[0]),
        "origin_px_y": float(origin_px[1]),
        "orientation_mode": orientation_mode,
        "orientation_score": orientation_score,
        "desired_start_x_mm": x0,
        "desired_start_y_mm": y0,
        "desired_end_x_mm": x1,
        "desired_end_y_mm": y1,
        "center_square_side_px_1": side_lengths[0],
        "center_square_side_px_2": side_lengths[1],
        "center_square_side_px_3": side_lengths[2],
        "center_square_side_px_4": side_lengths[3],
        "ghost_alpha": args.ghost_alpha,
        "data_source": "manual clicked robot centers",
        "frame_spacing_assumed": False,
        "rotation_start_frame": rotation_start_frame,
        "rotation_start_time_video_s": rotation_start_frame / fps,
    }
    pd.DataFrame([settings]).to_csv(out_dir / "analysis_settings.csv", index=False)

    print("\nDone. Saved:")
    print(f"  {out_dir / 'FIGURE_manual_ghost_clean.png'}")
    print(f"  {out_dir / 'FIGURE_final_overlay.png'}")
    print(f"  {out_dir / 'DATA_raw_manual_samples_px.csv'}")
    print(f"  {out_dir / 'DATA_clicked_centers_positions_velocity.csv'}")
    print(f"  {out_dir / 'DATA_clicked_segment_velocity.csv'}")
    print(f"  {out_dir / 'analysis_settings.csv'}")
    print(f"\nScale: {px_per_mm:.4f} px/mm")
    print(f"Orientation mode: {orientation_mode} (fixed, no manual direction click)")
    print("No fixed frame spacing was assumed.")
    print(f"Rotation start frame: {rotation_start_frame} -> time_s = 0 in CSV")


if __name__ == "__main__":
    main()
