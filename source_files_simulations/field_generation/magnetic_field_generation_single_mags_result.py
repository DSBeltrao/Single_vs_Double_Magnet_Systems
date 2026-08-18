import numpy as np
import pyvista as pv
import magpylib as magpy

# ============================================================
# PARAMETERS
# ============================================================

POOL_SIZE = 0.15          # 15 cm
GRID_N = 15               # 15 x 15 grid (1 cm spacing)
HALF = POOL_SIZE / 2

MAGNET_POS = np.array([0.0, 0.0, -0.205])  # 20.5 cm below pool
MAGNET_MOMENT = 1.0       # arbitrary (direction-only study)

ANGLE_STEP = 1            # magnet rotation step [deg]
THETA_STEP = 5            # sphere polar resolution
PHI_STEP = 5              # sphere azimuth resolution

# ============================================================
# GRID
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
    """Map unit vector to nearest (theta, phi) bin index"""
    x, y, z = u
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x) % (2*np.pi)

    it = int(theta / np.deg2rad(THETA_STEP)) % len(theta_vals)
    ip = int(phi   / np.deg2rad(PHI_STEP))   % len(phi_vals)
    return it, ip

# ============================================================
# MAGNET SETUP (Magpylib)
# ============================================================

magnet = magpy.magnet.Cuboid(
    magnetization=(0, 0, 1e6),  # realistic NdFeB order
    dimension=(0.01, 0.01, 0.01),
    position=MAGNET_POS
)

# ============================================================
# ROTATION HELPERS
# ============================================================

def rotate_magnet(mag, ax, ay):
    # Reset to initial orientation
    mag.orientation = None

    # Rotate around x-axis
    mag.rotate_from_angax(
        angle=ax,
        axis=[1, 0, 0],
        anchor=mag.position
    )

    # Rotate around y-axis
    mag.rotate_from_angax(
        angle=ay,
        axis=[0, 1, 0],
        anchor=mag.position
    )

# ============================================================
# COVERAGE COMPUTATION
# ============================================================

coverage_maps = []

for point in grid_points:
    reached = np.zeros((len(theta_vals), len(phi_vals)), dtype=bool)

    for ax in range(0, 360, ANGLE_STEP):
        for ay in range(0, 360, ANGLE_STEP):

            rotate_magnet(magnet, ax, ay)
            B = magpy.getB(magnet, point)

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
plotter.set_background("white")

sphere_radius = 0.004   # visual size only
scale = 1.3 * POOL_SIZE / GRID_N

for idx, point in enumerate(grid_points):

    reached = coverage_maps[idx]

    Theta, Phi = np.meshgrid(theta_vals, phi_vals, indexing="ij")
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    # scale & shift sphere to grid location
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

plotter.add_text(
    "Achievability of Target Directions Single Magnet System",
    position="upper_edge",
    font_size=14,
    color="black"
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
