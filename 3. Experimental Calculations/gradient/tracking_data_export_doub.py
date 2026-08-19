import cv2
import numpy as np
import csv

# --- Load homography ---
H = np.load("homography.npy")

video_path = "grad_doub\\2.0_vids\\doub_vid_1.mp4"
output_csv = "grad_doub\\2.0csv\\tracking_output_doub_1.csv"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

# --- Warp FIRST ---
frame = cv2.warpPerspective(frame, H, (1500, 800))

# --- Select ROI on warped frame ---
bbox = cv2.selectROI("Select MR", frame, False)
cv2.destroyWindow("Select MR")

# --- Tracker ---
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

positions = []
frame_id = 0

# Optional: frame skipping (set to 1 for full accuracy)
frame_skip = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_skip != 0:
        frame_id += 1
        continue

    # --- Warp every frame ---
    frame = cv2.warpPerspective(frame, H, (1500, 800))

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cx = x + w // 2
        cy = y + h // 2

        time = frame_id / fps
        positions.append([time, cx, cy])

        # --- Draw tracking ---
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    else:
        cv2.putText(frame, "Tracking failure", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- Show tracking window ---
    cv2.imshow("Tracking (Warped)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# --- Save to CSV ---
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "x_px", "y_px"])
    writer.writerows(positions)

print(f"Saved {len(positions)} points to {output_csv}")