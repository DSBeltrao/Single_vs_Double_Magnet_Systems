import cv2
import numpy as np

video_path = "grad_doub\\2.0_vids\\doub_vid_1.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read video")

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        cv2.circle(frame, (x, y), 5, (0,0,255), -1)
        cv2.imshow("Select 4 points", frame)

cv2.imshow("Select 4 points", frame)
cv2.setMouseCallback("Select 4 points", click_event)

print("Click 4 corners of a known rectangle (clockwise!)")

cv2.waitKey(0)
cv2.destroyAllWindows()

pts_src = np.array(points, dtype=float)

# Define real-world rectangle (you choose dimensions)
width = 1500   # arbitrary but proportional
height = 800

pts_dst = np.array([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
], dtype=float)

H, _ = cv2.findHomography(pts_src, pts_dst)

np.save("homography.npy", H)

print("Homography saved!")