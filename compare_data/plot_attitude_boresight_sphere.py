from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Geometry used by the current inspection_attitude_4phase.cpp problem.
SENSOR_BORESIGHT = np.array([0.0, 1.0, 0.0], dtype=float)
EXCLUSION_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=float)
PHASE2_TARGET = np.array([0.0, 0.0, -1.0], dtype=float)
PHASE4_TARGET = np.array([0.0, -1.0, 0.0], dtype=float)
EXCLUSION_HALF_ANGLE_DEG = 60.0
PHASE2_HALF_ANGLE_DEG = 20.0
PHASE4_HALF_ANGLE_DEG = 45.0


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    q_vec = np.array([qx, qy, qz], dtype=float)
    q_cross = np.array(
        [
            [0.0, -qz, qy],
            [qz, 0.0, -qx],
            [-qy, qx, 0.0],
        ],
        dtype=float,
    )
    return (qw**2 - q_vec @ q_vec) * np.eye(3) + 2.0 * np.outer(q_vec, q_vec) + 2.0 * qw * q_cross


def boresight_trajectory(quaternions: np.ndarray) -> np.ndarray:
    points = []
    for q in quaternions:
        q = q / np.linalg.norm(q)
        points.append(quaternion_to_rotation_matrix(q) @ SENSOR_BORESIGHT)
    return np.asarray(points)


def angle_to_direction(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    dots = np.clip(points @ direction, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def print_angle_consistency(state: pd.DataFrame, y_b: np.ndarray) -> None:
    checks = [
        ("angle_to_exclusion", EXCLUSION_DIRECTION),
        ("angle_to_los", PHASE2_TARGET),
        ("angle_to_docking", PHASE4_TARGET),
    ]
    for column, direction in checks:
        if column not in state.columns:
            continue
        expected = angle_to_direction(y_b, direction)
        recorded = state[column].to_numpy(dtype=float)
        max_error = float(np.nanmax(np.abs(expected - recorded)))
        print(f"{column}: max geometry/CSV angle mismatch = {max_error:.3e} deg")


def unit_sphere(resolution: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def spherical_cap(
    center_direction: np.ndarray,
    half_angle_deg: float,
    resolution: int = 56,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(center_direction, dtype=float)
    center = center / np.linalg.norm(center)
    half_angle = np.deg2rad(half_angle_deg)

    if abs(center[2]) < 0.9:
        v1 = np.cross(center, [0.0, 0.0, 1.0])
    else:
        v1 = np.cross(center, [1.0, 0.0, 0.0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(center, v1)

    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    phi = np.linspace(0.0, half_angle, max(8, resolution // 4))
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = np.zeros_like(theta_grid)
    y = np.zeros_like(theta_grid)
    z = np.zeros_like(theta_grid)
    for row in range(phi_grid.shape[0]):
        for col in range(phi_grid.shape[1]):
            rim_direction = v1 * np.cos(theta_grid[row, col]) + v2 * np.sin(theta_grid[row, col])
            point = center * np.cos(phi_grid[row, col]) + rim_direction * np.sin(phi_grid[row, col])
            x[row, col], y[row, col], z[row, col] = point
    return x, y, z


def sphere_z_circle(z_level: float, resolution: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if abs(z_level) > 1.0:
        return None
    radius = np.sqrt(1.0 - z_level**2)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    return radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z_level)


def sphere_y_circle(y_level: float, resolution: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if abs(y_level) > 1.0:
        return None
    radius = np.sqrt(1.0 - y_level**2)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    return radius * np.cos(theta), np.full_like(theta, y_level), radius * np.sin(theta)


def sphere_x_circle(x_level: float, resolution: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if abs(x_level) > 1.0:
        return None
    radius = np.sqrt(1.0 - x_level**2)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    return np.full_like(theta, x_level), radius * np.cos(theta), radius * np.sin(theta)


def draw_cap_boundary_guides(ax) -> None:
    sqrt3_half = np.sqrt(3.0) / 2.0

    for z_level in np.linspace(-1.0, -np.cos(np.deg2rad(PHASE2_HALF_ANGLE_DEG)), 5):
        circle = sphere_z_circle(float(z_level))
        if circle is not None:
            ax.plot(*circle, color="0.15", alpha=0.35, linewidth=0.8)

    for y_level in np.linspace(-1.0, -np.cos(np.deg2rad(PHASE4_HALF_ANGLE_DEG)), 5):
        circle = sphere_y_circle(float(y_level))
        if circle is not None:
            ax.plot(*circle, color="0.15", alpha=0.35, linewidth=0.8)

    for x_level in np.linspace(np.cos(np.deg2rad(EXCLUSION_HALF_ANGLE_DEG)), 1.0, 5):
        circle = sphere_x_circle(float(x_level))
        if circle is not None:
            ax.plot(*circle, color="0.15", alpha=0.35, linewidth=0.8)


def plot_phase_trajectory(ax, points: np.ndarray, phases: np.ndarray, phase: int, color: str, linewidth: float) -> None:
    idx = np.where(phases == phase)[0]
    if len(idx) < 2:
        return
    phase_points = points[idx]
    ax.plot(
        phase_points[:, 0],
        phase_points[:, 1],
        phase_points[:, 2],
        color=color,
        linewidth=linewidth,
        alpha=1.0,
    )


def make_plot(state_csv: Path, output: Path, dpi: int = 300) -> Path:
    state = pd.read_csv(state_csv)
    required = {"qw", "qx", "qy", "qz", "phase"}
    missing = required.difference(state.columns)
    if missing:
        raise ValueError(f"{state_csv} is missing required columns: {sorted(missing)}")

    quats = state[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    phases = state["phase"].to_numpy(dtype=int)
    y_b = boresight_trajectory(quats)
    print_angle_consistency(state, y_b)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(r"Trajectory of Sensor Boresight vector in frame $\mathcal{C}$", fontsize=24)
    ax = fig.add_subplot(111, projection="3d")

    sphere_x, sphere_y, sphere_z = unit_sphere()
    ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.08, color="lightgray", linewidth=0)
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.18, color="gray", linewidth=0.45)

    excl_x, excl_y, excl_z = spherical_cap(EXCLUSION_DIRECTION, EXCLUSION_HALF_ANGLE_DEG)
    ax.plot_surface(excl_x, excl_y, excl_z, color="yellow", alpha=0.42)

    p2_x, p2_y, p2_z = spherical_cap(PHASE2_TARGET, PHASE2_HALF_ANGLE_DEG)
    ax.plot_surface(p2_x, p2_y, p2_z, color="blue", alpha=0.32)

    p4_x, p4_y, p4_z = spherical_cap(PHASE4_TARGET, PHASE4_HALF_ANGLE_DEG)
    ax.plot_surface(p4_x, p4_y, p4_z, color="red", alpha=0.32)

    plot_phase_trajectory(ax, y_b, phases, phase=2, color="cyan", linewidth=3.5)
    plot_phase_trajectory(ax, y_b, phases, phase=4, color="black", linewidth=3.5)

    for phase, color, marker in [(2, "darkblue", "s"), (4, "darkred", "s")]:
        idx = np.where(phases == phase)[0]
        if len(idx):
            point = y_b[idx[-1]]
            ax.scatter(point[0], point[1], point[2], color=color, s=70, marker=marker, edgecolors=color)

    start = y_b[0]
    ax.scatter(start[0], start[1], start[2], color="black", s=60, marker="o", edgecolors="black")

    draw_cap_boundary_guides(ax)

    arrow_length = 1.2
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color="red", arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color="green", arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color="blue", arrow_length_ratio=0.1, linewidth=2)
    ax.text(arrow_length * 1.1, 0, 0, "X", color="red", fontsize=13, weight="bold")
    ax.text(0, arrow_length * 1.1, 0, "Y", color="green", fontsize=13, weight="bold")
    ax.text(0, 0, arrow_length * 1.1, "Z", color="blue", fontsize=13, weight="bold")

    legend_elements = [
        mpatches.Patch(color="yellow", alpha=0.6, label=f"Exclusion Zone ({EXCLUSION_HALF_ANGLE_DEG:.0f} deg)"),
        mpatches.Patch(color="blue", alpha=0.6, label=f"Phase 2 Target Zone ({PHASE2_HALF_ANGLE_DEG:.0f} deg)"),
        mpatches.Patch(color="red", alpha=0.6, label=f"Phase 4 Target Zone ({PHASE4_HALF_ANGLE_DEG:.0f} deg)"),
        mlines.Line2D([], [], color="cyan", linewidth=3.5, label="Phase 2 Trajectory"),
        mlines.Line2D([], [], color="black", linewidth=3.5, label="Phase 4 Trajectory"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.02, 0.78), fontsize=11)

    ax.view_init(elev=39, azim=-31)
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot attitude boresight trajectory on the unit sphere.")
    parser.add_argument(
        "--state-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "attitude" / "4phase_attitude_state_ver1.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "attitude_boresight_sphere_ver1_corrected.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = make_plot(args.state_csv, args.output, dpi=args.dpi)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
