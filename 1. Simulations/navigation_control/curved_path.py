import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# GIVEN PATH (UNCHANGED, NOW IN 3D)
# ============================================================
pool_size = 15.0

segment_lengths = [0.5, 1, 1.7, 2.8, 4.3, 6.2, 7.3, 7, 8, 8, 7.5, 7, 7, 6, 5, 4, 4, 3]
turn_angles_deg = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
points_per_segment = 50

pos = np.array([6.0, 6.0, 0.0])
path_points = [pos.copy()]
theta = np.pi

for L, turn_deg in zip(segment_lengths, turn_angles_deg):
    dir_vec = np.array([np.cos(theta), np.sin(theta), 0.0])
    s = np.linspace(0, L, points_per_segment)
    for ds in s[1:]:
        path_points.append(pos + ds * dir_vec)
    pos = path_points[-1]
    theta -= np.deg2rad(turn_deg)

path = np.array(path_points)

# ============================================================
# 1. TANGENT
# ============================================================
dpath = np.gradient(path, axis=0)
T = dpath / np.linalg.norm(dpath, axis=1, keepdims=True)

# ============================================================
# 2. BISHOP FRAME (PARALLEL TRANSPORT)
# ============================================================
N = np.zeros_like(T)
B = np.zeros_like(T)

# Initial normal
ref = np.array([0, 0, 1])
if abs(np.dot(ref, T[0])) > 0.9:
    ref = np.array([1, 0, 0])

N[0] = np.cross(T[0], ref)
N[0] /= np.linalg.norm(N[0])
B[0] = np.cross(T[0], N[0])

for i in range(1, len(T)):
    v = T[i] + T[i-1]
    if np.linalg.norm(v) < 1e-6:
        N[i] = N[i-1]
        B[i] = B[i-1]
        continue

    v /= np.linalg.norm(v)
    N[i] = N[i-1] - np.dot(N[i-1], v) * v
    N[i] /= np.linalg.norm(N[i])
    B[i] = np.cross(T[i], N[i])

# ============================================================
# 3. LOCAL HELIX (MF DIRECTION)
# ============================================================
helix_radius = 0.3   # cm
helix_pitch = 1.5    # cm per turn

arc_len = np.zeros(len(path))
arc_len[1:] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
phase = 2 * np.pi * arc_len / helix_pitch

MF = np.cos(phase)[:, None] * N + np.sin(phase)[:, None] * B
helix_points = path + helix_radius * MF

# ============================================================
# 4. 3D VISUALIZATION
# ============================================================
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Centerline
ax.plot(
    path[:,0], path[:,1], path[:,2],
    color='black', linewidth=2, label='Centerline'
)

# Helix
ax.plot(
    helix_points[:,0], helix_points[:,1], helix_points[:,2],
    color='red', linewidth=1.5, label='MF helix'
)

# Bishop frame vectors
skip = 150
scale = 0.8

ax.quiver(
    path[::skip,0], path[::skip,1], path[::skip,2],
    N[::skip,0], N[::skip,1], N[::skip,2],
    color='blue', length=scale, label='N'
)

ax.quiver(
    path[::skip,0], path[::skip,1], path[::skip,2],
    B[::skip,0], B[::skip,1], B[::skip,2],
    color='green', length=scale, label='B'
)

ax.set_xlim(0, pool_size)
ax.set_ylim(0, pool_size)
ax.set_zlim(-2, 2)

ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
ax.set_zlabel("Z [cm]")
ax.set_title("3D Cochlear Path with Bishop Frame and MF Helix")

ax.legend()
plt.show()
