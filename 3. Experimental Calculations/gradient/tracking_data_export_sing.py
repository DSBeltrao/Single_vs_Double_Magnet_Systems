import cv2
import numpy as np
import csv

# ============================================================
# SETTINGS
# ============================================================

video_path = "grad_sing\\sing_vid_7.mp4"
output_csv = "grad_sing\\tracking_output_7.csv"
preprocess_file = "grad_sing\\single_preprocess.npy"

# Initial resize of full frame
# Keep 1.0 for full resolution, 0.5 for faster ROI selection
scale = 0.5

# Zoom applied AFTER cropping
# This gives the MR more pixels during tracking
zoom = 2.0

# Process every nth frame
frame_skip = 3

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

# ============================================================
# PREPROCESS FIRST FRAME
# ============================================================

frame_scaled = cv2.resize(frame, None, fx=scale, fy=scale)

# Select larger analysis ROI first
crop_box = cv2.selectROI("Select analysis ROI", frame_scaled, False)
cv2.destroyWindow("Select analysis ROI")

x0, y0, w0, h0 = [int(v) for v in crop_box]

if w0 == 0 or h0 == 0:
    print("No ROI selected.")
    cap.release()
    exit()

# Save preprocessing parameters for calibration/speed script
np.save(preprocess_file, np.array([x0, y0, w0, h0, scale, zoom], dtype=float))

print("Saved preprocessing parameters:")
print(f"x0={x0}, y0={y0}, w0={w0}, h0={h0}, scale={scale}, zoom={zoom}")

# Crop and zoom first frame
frame_crop = frame_scaled[y0:y0+h0, x0:x0+w0]
frame_zoom = cv2.resize(
    frame_crop,
    None,
    fx=zoom,
    fy=zoom,
    interpolation=cv2.INTER_CUBIC
)

# Select MR inside zoomed ROI
bbox = cv2.selectROI("Select MR inside zoomed ROI", frame_zoom, False)
cv2.destroyWindow("Select MR inside zoomed ROI")

if bbox[2] == 0 or bbox[3] == 0:
    print("No MR selected.")
    cap.release()
    exit()

# ============================================================
# TRACKER
# ============================================================

tracker = cv2.TrackerCSRT_create()
tracker.init(frame_zoom, bbox)

positions = []
frame_id = 0

# ============================================================
# TRACK LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_skip != 0:
        frame_id += 1
        continue

    frame_scaled = cv2.resize(frame, None, fx=scale, fy=scale)

    # Apply same crop
    frame_crop = frame_scaled[y0:y0+h0, x0:x0+w0]

    # Apply same zoom
    frame_zoom = cv2.resize(
        frame_crop,
        None,
        fx=zoom,
        fy=zoom,
        interpolation=cv2.INTER_CUBIC
    )

    success, bbox = tracker.update(frame_zoom)

    if success:
        x, y, w, h = [int(v) for v in bbox]

        cx = x + w // 2
        cy = y + h // 2

        time_s = frame_id / fps

        # Coordinates are in ZOOMED-CROP coordinate system
        positions.append([time_s, cx, cy])

        cv2.rectangle(frame_zoom, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame_zoom, (cx, cy), 3, (0, 0, 255), -1)

        cv2.putText(
            frame_zoom,
            f"t = {time_s:.2f} s, pos = ({cx}, {cy})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    else:
        cv2.putText(
            frame_zoom,
            "Tracking failure",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Tracking - cropped and zoomed", frame_zoom)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

# ============================================================
# SAVE CSV
# ============================================================

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "x_px", "y_px"])
    writer.writerows(positions)

print(f"Saved {len(positions)} points to {output_csv}")
print(f"Saved preprocessing info to {preprocess_file}")