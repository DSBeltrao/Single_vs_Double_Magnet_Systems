import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
R = 0.05                 # helix radius
rotations = 6
N = 400

phi = np.linspace(0, 2*np.pi*rotations, N)

# propulsion gain (1 cm per 3 rotations)
alpha = 0.01 / (3 * 2*np.pi)

# -----------------------------
# Straight path
# -----------------------------
p = np.zeros((N, 3))
p[:, 0] = alpha * phi    # motion along x

# -----------------------------
# Helix vectors (PURE rotation)
# -----------------------------
v = np.zeros((N, 3))
v[:, 1] = R * np.cos(phi)
v[:, 2] = R * np.sin(phi)

# -----------------------------
# Helix curve
# -----------------------------
helix = p + v

# -----------------------------
# Sanity check (must be zero)
# -----------------------------
d_hat = np.array([1.0, 0.0, 0.0])
dot = v @ d_hat
print("Max |d · v| =", np.max(np.abs(dot)))

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection="3d")

ax.plot(p[:,0], p[:,1], p[:,2],
        'k--', lw=2, label="Path")

ax.plot(helix[:,0], helix[:,1], helix[:,2],
        'r', lw=1.5, label="Helix")

idx = np.arange(0, N, 15)
ax.quiver(p[idx,0], p[idx,1], p[idx,2],
          v[idx,0], v[idx,1], v[idx,2],
          color="tab:blue",
          arrow_length_ratio=0.2,
          linewidth=1,
          label="Rotating vectors")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 0.4, 0.4])
ax.legend()
plt.tight_layout()
plt.show()
