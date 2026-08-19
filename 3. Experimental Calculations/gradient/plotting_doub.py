import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

video_path = "grad_doub\\2.0_vids\\doub_vid_5.mp4"
csv_path = "grad_doub\\2.0csv\\tracking_output_doub_5.csv"

homography_path = "homography.npy"

warp_width = 1500
warp_height = 800

square_size_mm = 10.0   # clicked square is 1 cm = 10 mm
window_s = 2.0          # velocity estimation window

save_plot = True
output_plot = "speed_vs_distance_double_5.pdf"
output_csv = "grad_doub\\2.0csv\\speed_vs_distance_double_5.csv"

# ============================================================
# LOAD HOMOGRAPHY
# ============================================================

H = np.load(homography_path)

# ============================================================
# CLICK 4 CORNERS OF 1 cm SQUARE ON WARPED FRAME
# ============================================================

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read video frame.")

# Apply SAME warp as tracking
frame = cv2.warpPerspective(frame, H, (warp_width, warp_height))

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(len(points)),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow("Calibration - warped view", frame)
        print(f"Point {len(points)}: ({x}, {y})")

cv2.imshow("Calibration - warped view", frame)
cv2.setMouseCallback("Calibration - warped view", click_event)

print("Click four corners of a real 1 cm square:")
print("1) top-left")
print("2) top-right")
print("3) bottom-right")
print("4) bottom-left")
print("Press any key after selecting all four points.")

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    raise RuntimeError("Exactly 4 points are required.")

pts = np.array(points, dtype=float)

top_left = pts[0]
top_right = pts[1]
bottom_right = pts[2]
bottom_left = pts[3]

# Origin = center of clicked square
origin = np.mean(pts, axis=0)

# Average side length of square
side_top = np.linalg.norm(top_right - top_left)
side_right = np.linalg.norm(bottom_right - top_right)
side_bottom = np.linalg.norm(bottom_left - bottom_right)
side_left = np.linalg.norm(top_left - bottom_left)

one_cm_px = np.mean([side_top, side_right, side_bottom, side_left])
mm_per_pixel = square_size_mm / one_cm_px

print("\nCalibration results:")
print(f"Origin = ({origin[0]:.2f}, {origin[1]:.2f}) px")
print(f"1 cm = {one_cm_px:.2f} px")
print(f"Scale = {mm_per_pixel:.5f} mm/px")

# ============================================================
# LOAD TRACKING CSV
# ============================================================

data = pd.read_csv(csv_path)

time_s = data["time_s"].to_numpy()
x_px = data["x_px"].to_numpy()
y_px = data["y_px"].to_numpy()

positions = np.column_stack((x_px, y_px))

# ============================================================
# DISTANCE FROM ORIGIN
# ============================================================

dist_px = np.linalg.norm(positions - origin, axis=1)
dist_mm = dist_px * mm_per_pixel
dist_cm = dist_mm / 10.0

# ============================================================
# VELOCITY USING SLIDING-WINDOW LINEAR FIT
# ============================================================

def sliding_velocity(time_s, distance_mm, window_s=2.0, min_points=5):
    speed_mm_s = []
    distance_cm_mid = []
    time_mid = []

    half_window = window_s / 2.0

    for i, t in enumerate(time_s):
        mask = (time_s >= t - half_window) & (time_s <= t + half_window)

        if np.sum(mask) < min_points:
            continue

        t_window = time_s[mask]
        d_window = distance_mm[mask]

        # distance = velocity*time + offset
        coeffs = np.polyfit(t_window, d_window, 1)

        # Use magnitude of speed
        v = abs(coeffs[0])

        speed_mm_s.append(v)
        distance_cm_mid.append(distance_mm[i] / 10.0)
        time_mid.append(t)

    return (
        np.array(time_mid),
        np.array(distance_cm_mid),
        np.array(speed_mm_s)
    )

time_plot, dist_cm_plot, speed_mm_s = sliding_velocity(
    time_s,
    dist_mm,
    window_s=window_s
)

# ============================================================
# SAVE SPEED VS DISTANCE CSV
# ============================================================

output_data = pd.DataFrame({
    "time_s": time_plot,
    "distance_cm": dist_cm_plot,
    "speed_mm_s": speed_mm_s
})

output_data.to_csv(output_csv, index=False)

print(f"\nSaved speed-distance CSV as: {output_csv}")
print(f"Number of data points saved: {len(output_data)}")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    dist_cm_plot,
    speed_mm_s,
    s=6,
    color="k",
    linewidths=0
)

#ax.plot(
#    x_fit,
#    y_fit,
#    color="k",
#    linewidth=1
#)

ax.set_xlabel("Distance / cm")
ax.set_ylabel("Speed / mm/s")

ax.tick_params(which="both", top=True, right=True)

fig.tight_layout()

if save_plot:
    plt.savefig(output_plot, format="pdf")
    print(f"\nSaved plot as: {output_plot}")

plt.show()