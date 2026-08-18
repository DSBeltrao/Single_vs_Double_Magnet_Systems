import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# =========================
# SETTINGS (match your PyVista)
# =========================
POOL_SIZE = 0.15
HALF = POOL_SIZE/2

Z_TOP = +0.125
Z_BOTTOM = -0.125

ANGLE_STEP = 5      # use 1 for exact match, 2/3/5 for faster testing
THETA_STEP = 5
PHI_STEP = 5

sphere_radius = 0.015  # purely for visualization

cone_deg = 5.0        # for "reachable within cone" when stepping phase
cos_thresh = np.cos(np.deg2rad(cone_deg))

# =========================
# ONE POINT (ROI centered at 0 like your PyVista)
# =========================
point = np.array([0.0, 0.04, 0.0])   # meters. change this

# =========================
# Sphere discretization + binning (same as your code)
# =========================
theta_vals = np.deg2rad(np.arange(0, 180, THETA_STEP))
phi_vals   = np.deg2rad(np.arange(0, 360, PHI_STEP))

def dir_to_bin(u):
    x, y, z = u
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x) % (2*np.pi)

    it = int(theta / np.deg2rad(THETA_STEP)) % len(theta_vals)
    ip = int(phi   / np.deg2rad(PHI_STEP))   % len(phi_vals)
    return it, ip

# =========================
# Magnets (same as your code)
# =========================
mag_top = magpy.magnet.Cuboid(
    magnetization=(0, 0, 1e6),
    dimension=(0.01, 0.01, 0.01),
    position=(0, 0, Z_TOP)
)

mag_bottom = magpy.magnet.Cuboid(
    magnetization=(0, 0, 1e6),
    dimension=(0.01, 0.01, 0.01),
    position=(0, 0, Z_BOTTOM)
)

def rotate_top(angle_deg):
    mag_top.orientation = None
    mag_top.rotate_from_angax(angle=angle_deg, axis=[1, 0, 0], anchor=mag_top.position)

def rotate_bottom(angle_deg):
    mag_bottom.orientation = None
    mag_bottom.rotate_from_angax(angle=angle_deg, axis=[0, 1, 0], anchor=mag_bottom.position)

# =========================
# Compute reachability map (same logic as PyVista)
# =========================
reached = np.zeros((len(theta_vals), len(phi_vals)), dtype=bool)

for a in range(0, 360, ANGLE_STEP):
    rotate_top(a)
    for b in range(0, 360, ANGLE_STEP):
        rotate_bottom(b)

        B = magpy.getB(mag_top, point) + magpy.getB(mag_bottom, point)
        n = np.linalg.norm(B)
        if n < 1e-12:
            continue

        u = B / n
        it, ip = dir_to_bin(u)
        reached[it, ip] = True

# =========================
# Helpers for plane stepping fallback
# =========================
def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v/n

def plane_basis_from_normal(n_plane):
    n = normalize(n_plane)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    N = np.cross(n, tmp); N = normalize(N)
    B = np.cross(n, N);   B = normalize(B)
    return N, B

def dir_on_plane(N, B, phi):
    return np.cos(phi)*N + np.sin(phi)*B

def best_alignment_with_reached(u):
    """
    Compute best dot-product between u and any REACHED bin direction
    (approx by checking bin centers).
    """
    u = normalize(u)
    # build bin-center directions once per call (fine for small grids)
    best = -1.0
    for it, th in enumerate(theta_vals):
        for ip, ph in enumerate(phi_vals):
            if not reached[it, ip]:
                continue
            v = np.array([np.cos(ph)*np.sin(th), np.sin(ph)*np.sin(th), np.cos(th)])
            best = max(best, float(np.dot(u, v)))
    return best

def first_reachable_by_phase(N, B, phi_des, rot_dir, dphi_deg):
    dphi = np.deg2rad(dphi_deg)
    for k in range(0, int(np.ceil(360/dphi_deg))+1):
        phi_try = phi_des + rot_dir*k*dphi
        v_try = dir_on_plane(N, B, phi_try)
        if best_alignment_with_reached(v_try) >= cos_thresh:
            return phi_try, v_try, k
    return None, None, None

# =========================
# Choose plane + vectors to test
# =========================
plane_normal = np.array([1.0, 0.0, 0.0])  # e.g. tangent along +x => plane is YZ
Nhat, Bhat = plane_basis_from_normal(plane_normal)

phi_current_deg = 20.0
phi_desired_deg = 130.0  # choose a "bad" one visually, then rerun
rot_dir = +1             # +1 CCW, -1 CW (in N-B parameterization)
dphi_deg = 5.0

v_current = dir_on_plane(Nhat, Bhat, np.deg2rad(phi_current_deg))
v_desired = dir_on_plane(Nhat, Bhat, np.deg2rad(phi_desired_deg))

phi_res, v_res, steps = first_reachable_by_phase(Nhat, Bhat, np.deg2rad(phi_desired_deg), rot_dir, dphi_deg)

print("\n=== SINGLE-SPHERE COHERENT TEST ===")
print("point [m] =", point)
print("ANGLE_STEP =", ANGLE_STEP, "deg")
print("desired best align =", best_alignment_with_reached(v_desired))
if v_res is None:
    print("No reachable direction found in 360° sweep.")
else:
    print(f"result phi = {(np.rad2deg(phi_res)%360):.2f} deg (steps={steps})")
    print("result best align =", best_alignment_with_reached(v_res))

# =========================
# Matplotlib sphere visualization (white/black bins)
# =========================
Theta, Phi = np.meshgrid(theta_vals, phi_vals, indexing="ij")
X = np.sin(Theta)*np.cos(Phi)
Y = np.sin(Theta)*np.sin(Phi)
Z = np.cos(Theta)

# scale and center at origin
Xv = sphere_radius*X
Yv = sphere_radius*Y
Zv = sphere_radius*Z

# facecolors from reached
facecolors = np.zeros((*reached.shape, 4))
facecolors[reached] = (1, 1, 1, 0.85)
facecolors[~reached] = (0, 0, 0, 0.85)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Reachable direction bins (coherent with PyVista logic)")

ax.plot_surface(Xv, Yv, Zv, facecolors=facecolors, linewidth=0, antialiased=False, shade=False)

def draw_vec(v, color, label):
    v = normalize(v)
    ax.quiver(0,0,0, v[0],v[1],v[2], length=sphere_radius*1.2, color=color, linewidth=2)
    ax.text(v[0]*sphere_radius*1.35, v[1]*sphere_radius*1.35, v[2]*sphere_radius*1.35, label, color=color)

draw_vec(v_current, "green", "current")
draw_vec(v_desired, "red", "desired")
if v_res is not None:
    draw_vec(v_res, "blue", "result")

ax.set_box_aspect([1,1,1])
lim = sphere_radius*1.5
ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.tight_layout()
plt.show()