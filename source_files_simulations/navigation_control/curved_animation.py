import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# PATH (your cochlea, unchanged)
# ============================================================
pool_size = 15.0

segment_lengths = [0.5, 1, 1.7, 2.8, 4.3, 6.2, 7.3, 7, 8, 8, 7.5, 7, 7, 6, 5, 4, 4, 3]
turn_angles_deg = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
points_per_segment = 50

pos = np.array([6.0, 6.0, 0.0])
path_points = [pos.copy()]
theta = np.pi

corner_indices = []
idx = 0

for L, turn_deg in zip(segment_lengths, turn_angles_deg):
    dir_vec = np.array([np.cos(theta), np.sin(theta), 0.0])
    s = np.linspace(0, L, points_per_segment)
    for ds in s[1:]:
        path_points.append(pos + ds * dir_vec)
    pos = path_points[-1]
    theta -= np.deg2rad(turn_deg)
    idx += points_per_segment - 1
    corner_indices.append(idx)

path = np.array(path_points)

# ============================================================
# TANGENT
# ============================================================
dpath = np.gradient(path, axis=0)
T = dpath / np.linalg.norm(dpath, axis=1, keepdims=True)

# ============================================================
# BISHOP FRAME
# ============================================================
N = np.zeros_like(T)
B = np.zeros_like(T)

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
# HELIX PHASE (propulsion)
# ============================================================
helix_pitch = 1.5  # cm per rotation
arc_len = np.zeros(len(path))
arc_len[1:] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
phase = 2 * np.pi * arc_len / helix_pitch
MF_prop = np.cos(phase)[:,None]*N + np.sin(phase)[:,None]*B

# ============================================================
# BUILD ANIMATION TIMELINE (with reorientation)
# ============================================================
path_anim = []
MF_anim   = []

align_frames = 20

corner_set = set(corner_indices)

for i in range(len(path)-1):

    # --- normal propulsion step ---
    path_anim.append(path[i])
    MF_anim.append(MF_prop[i])

    # --- reorientation at corner ---
    if i in corner_set:
        d_new = T[i+1]

        align_vec = np.cross([0,0,1], d_new)
        if np.linalg.norm(align_vec) < 1e-6:
            align_vec = np.cross([1,0,0], d_new)
        align_vec /= np.linalg.norm(align_vec)

        v_start = MF_prop[i]

        for s in np.linspace(0, 1, align_frames):
            v = (1-s)*v_start + s*align_vec
            v /= np.linalg.norm(v)
            path_anim.append(path[i])   # fixed position
            MF_anim.append(v)

path_anim = np.array(path_anim)
MF_anim   = np.array(MF_anim)

# ============================================================
# ANIMATION
# ============================================================
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(path[:,0], path[:,1], path[:,2], 'k', linewidth=1)
ax.set_xlim(0, pool_size)
ax.set_ylim(0, pool_size)
ax.set_zlim(-2, 2)

ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
ax.set_zlabel("Z [cm]")
ax.set_title("Microrobot with Stop-and-Align Control")

vec_len = 1.0
quiver = ax.quiver(
    path_anim[0,0], path_anim[0,1], path_anim[0,2],
    MF_anim[0,0], MF_anim[0,1], MF_anim[0,2],
    color='red', length=vec_len
)

def update(i):
    global quiver
    quiver.remove()
    quiver = ax.quiver(
        path_anim[i,0], path_anim[i,1], path_anim[i,2],
        MF_anim[i,0], MF_anim[i,1], MF_anim[i,2],
        color='red', length=vec_len
    )
    return quiver,

ani = FuncAnimation(fig, update, frames=len(path_anim), interval=30)
plt.show()