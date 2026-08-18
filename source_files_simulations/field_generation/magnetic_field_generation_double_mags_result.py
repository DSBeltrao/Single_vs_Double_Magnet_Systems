import numpy as np
import pyvista as pv
import magpylib as magpy

pv.global_theme.allow_empty_mesh = True

# ============================================================
# PARAMETERS
# ============================================================

POOL_SIZE = 0.15        # 15 cm pool
GRID_N = 15
HALF = POOL_SIZE / 2

Z_TOP = +0.205           # top magnet position
Z_BOTTOM = -0.205        # bottom magnet position

ANGLE_STEP = 1          # degrees
THETA_STEP = 5          # sphere discretization
PHI_STEP = 5

# ============================================================
# GRID (xy-plane at z = 0)
# ============================================================

xs = np.linspace(-HALF, HALF, GRID_N)
ys = np.linspace(-HALF, HALF, GRID_N)

grid_points = [
    np.array([x, y, 0.0])
    for x in xs for y in ys
]

# ============================================================
# SPHERE DISCRETIZATION
# ============================================================

theta_vals = np.deg2rad(np.arange(0, 180, THETA_STEP))
phi_vals   = np.deg2rad(np.arange(0, 360, PHI_STEP))

def dir_to_bin(u):
    x, y, z = u
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x) % (2*np.pi)

    it = int(theta / np.deg2rad(THETA_STEP)) % len(theta_vals)
    ip = int(phi   / np.deg2rad(PHI_STEP))   % len(phi_vals)
    return it, ip

# ============================================================
# MAGNET SETUP
# ============================================================

mag_top = magpy.magnet.Cuboid(
    magnetization=(0, 0, 1e6),   # +z
    dimension=(0.01, 0.01, 0.01),
    position=(0, 0, Z_TOP)
)

mag_bottom = magpy.magnet.Cuboid(
    magnetization=(0, 0, 1e6),   # +z
    dimension=(0.01, 0.01, 0.01),
    position=(0, 0, Z_BOTTOM)
)

# ============================================================
# ROTATION HELPERS
# ============================================================

def rotate_top(mag, angle_deg):
    mag.orientation = None
    mag.rotate_from_angax(
        angle=angle_deg,
        axis=[1, 0, 0],          # x-axis
        anchor=mag.position
    )

def rotate_bottom(mag, angle_deg):
    mag.orientation = None
    mag.rotate_from_angax(
        angle=angle_deg,
        axis=[0, 1, 0],          # y-axis
        anchor=mag.position
    )

# ============================================================
# COVERAGE COMPUTATION
# ============================================================

coverage_maps = []

for point in grid_points:
    reached = np.zeros((len(theta_vals), len(phi_vals)), dtype=bool)

    for a in range(0, 360, ANGLE_STEP):
        rotate_top(mag_top, a)

        for b in range(0, 360, ANGLE_STEP):
            rotate_bottom(mag_bottom, b)

            B = (
                magpy.getB(mag_top, point)
                + magpy.getB(mag_bottom, point)
            )

            norm = np.linalg.norm(B)
            if norm < 1e-12:
                continue

            u = B / norm
            it, ip = dir_to_bin(u)
            reached[it, ip] = True

    coverage_maps.append(reached)

# ============================================================
# PYVISTA VISUALIZATION
# ============================================================

plotter = pv.Plotter()
plotter.disable_picking()
plotter.set_background("white")

sphere_radius = 0.004

for idx, point in enumerate(grid_points):

    reached = coverage_maps[idx]

    Theta, Phi = np.meshgrid(theta_vals, phi_vals, indexing="ij")
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    X = sphere_radius * X + point[0]
    Y = sphere_radius * Y + point[1]
    Z = sphere_radius * Z + point[2]

    grid = pv.StructuredGrid(X, Y, Z)
    surf = grid.extract_surface()
    surf["coverage"] = reached.flatten().astype(float)

    plotter.add_mesh(
        surf,
        scalars="coverage",
        cmap=["black", "white"],
        clim=[0, 1],
        interpolate_before_map=False,
        show_edges=False
    )

plotter.add_axes(
    color="black",
    xlabel='X ',
    ylabel='Y',
    zlabel='Z'
)

pool = pv.Plane(
    center=(0.0, 0.0, 0.0),
    direction=(0, 0, 1),
    i_size=POOL_SIZE,
    j_size=POOL_SIZE,
    i_resolution=1,
    j_resolution=1
)

plotter.add_mesh(
    pool,
    style="wireframe",
    color="black",
    line_width=2,
)

plotter.add_text(
    "Workspace: 15 cm × 15 cm",
    position="lower_right",
    font_size=12,
)

plotter.remove_scalar_bar()
plotter.show()
