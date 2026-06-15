from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "mathtext.fontset": "dejavuserif",
        "savefig.dpi": 600,
    }
)

BORESIGHT_BODY = np.array([0.0, 1.0, 0.0], dtype=float)
SUN_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=float)
C_N_LOS = np.array([0.0, 0.0, 1.0], dtype=float)
C_N_DOCK = np.array([0.0, 1.0, 0.0], dtype=float)

ALPHA_DEG = 60.0
THETA_LOS_DEG = 20.0
THETA_DOCK_DEG = 45.0

PHASE2_CENTER = -C_N_LOS
PHASE4_CENTER = -C_N_DOCK
SUN_HALF_ANGLE_DEG = ALPHA_DEG
PHASE2_HALF_ANGLE_DEG = ALPHA_DEG - THETA_LOS_DEG
PHASE4_HALF_ANGLE_DEG = ALPHA_DEG - THETA_DOCK_DEG
AZ_MIN = -1.5 * np.pi
AZ_MAX = 0.5 * np.pi

PHASE_STYLES = {
    1: {"color": "#6f6f6f", "label": "Phase 1"},
    2: {"color": "#0072B2", "label": "Phase 2"},
    3: {"color": "#009E73", "label": "Phase 3"},
    4: {"color": "#D55E00", "label": "Phase 4"},
}
SOLVER_STYLES = {
    "DDP": {
        "linestyle": "-",
        "linewidth": 1.2,
        "alpha": 0.72,
        "zorder": 4,
    },
    "Rockit/IPOPT": {
        "linestyle": (0, (3.0, 2.0)),
        "linewidth": 1.35,
        "alpha": 1.0,
        "zorder": 6,
    },
}


@dataclass(frozen=True)
class AttitudeCase:
    name: str
    state: pd.DataFrame
    boresight: np.ndarray
    phases: np.ndarray


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def q_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def boresight_from_state(state: pd.DataFrame) -> np.ndarray:
    quats = state[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    return np.array([q_to_rotation_matrix(q) @ BORESIGHT_BODY for q in quats])


def load_ddp(state_csv: Path) -> AttitudeCase:
    state = pd.read_csv(state_csv).sort_values(["phase", "time"]).reset_index(drop=True)
    check_columns(state, state_csv)
    return AttitudeCase("DDP", state, boresight_from_state(state), state["phase"].to_numpy(dtype=int))


def load_rockit(rotational_csv: Path) -> AttitudeCase:
    state = pd.read_csv(rotational_csv).sort_values(["phase", "time", "node"]).reset_index(drop=True)
    check_columns(state, rotational_csv)
    return AttitudeCase("Rockit/IPOPT", state, boresight_from_state(state), state["phase"].to_numpy(dtype=int))


def check_columns(state: pd.DataFrame, csv_path: Path) -> None:
    required = {"qw", "qx", "qy", "qz", "phase"}
    missing = required.difference(state.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")


def to_az_el(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    az = wrap_azimuth(np.arctan2(y, x))
    el = np.arcsin(np.clip(z, -1.0, 1.0))
    return az, el


def wrap_azimuth(az: np.ndarray) -> np.ndarray:
    return (az - AZ_MIN) % (2.0 * np.pi) + AZ_MIN


def unit_vectors_from_az_el(az_grid: np.ndarray, el_grid: np.ndarray) -> np.ndarray:
    cos_el = np.cos(el_grid)
    return np.stack(
        [cos_el * np.cos(az_grid), cos_el * np.sin(az_grid), np.sin(el_grid)],
        axis=-1,
    )


def unit_sphere(resolution: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    return (
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
    )


def spherical_cap(center_direction: np.ndarray, half_angle_deg: float, resolution: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = normalize(center_direction)
    half_angle = np.deg2rad(half_angle_deg)
    if abs(center[2]) < 0.9:
        basis_1 = normalize(np.cross(center, [0.0, 0.0, 1.0]))
    else:
        basis_1 = normalize(np.cross(center, [1.0, 0.0, 0.0]))
    basis_2 = np.cross(center, basis_1)

    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    phi = np.linspace(0.0, half_angle, max(10, resolution // 4))
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    points = (
        center[None, None, :] * np.cos(phi_grid[..., None])
        + (basis_1[None, None, :] * np.cos(theta_grid[..., None]) + basis_2[None, None, :] * np.sin(theta_grid[..., None]))
        * np.sin(phi_grid[..., None])
    )
    return points[..., 0], points[..., 1], points[..., 2]


def cone_side(center_direction: np.ndarray, half_angle_deg: float, resolution: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = normalize(center_direction)
    half_angle = np.deg2rad(half_angle_deg)
    if abs(center[2]) < 0.9:
        basis_1 = normalize(np.cross(center, [0.0, 0.0, 1.0]))
    else:
        basis_1 = normalize(np.cross(center, [1.0, 0.0, 0.0]))
    basis_2 = np.cross(center, basis_1)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    radius = np.linspace(0.0, 1.0, 18)
    theta_grid, radius_grid = np.meshgrid(theta, radius)
    rim = center[None, None, :] * np.cos(half_angle) + (
        basis_1[None, None, :] * np.cos(theta_grid[..., None]) + basis_2[None, None, :] * np.sin(theta_grid[..., None])
    ) * np.sin(half_angle)
    points = radius_grid[..., None] * rim
    return points[..., 0], points[..., 1], points[..., 2]


def plot_trajectory_2d(ax, case: AttitudeCase, solver_label: str | None = None) -> None:
    az, el = to_az_el(case.boresight)
    for phase in sorted(PHASE_STYLES):
        mask = case.phases == phase
        if np.count_nonzero(mask) < 2:
            continue
        phase_az = az[mask].copy()
        phase_el = el[mask].copy()
        jumps = np.where(np.abs(np.diff(phase_az)) > np.pi)[0]
        if len(jumps):
            phase_az = np.insert(phase_az, jumps + 1, np.nan)
            phase_el = np.insert(phase_el, jumps + 1, np.nan)
        phase_style = PHASE_STYLES[phase]
        solver_style = SOLVER_STYLES[case.name]
        ax.plot(
            phase_az,
            phase_el,
            color=phase_style["color"],
            linestyle=solver_style["linestyle"],
            linewidth=solver_style["linewidth"],
            alpha=solver_style["alpha"],
            zorder=solver_style["zorder"],
        )
    ax.scatter(az[0], el[0], s=13, marker="o", color="white", edgecolor="black", linewidth=0.65, zorder=8, label=None)
    ax.scatter(az[-1], el[-1], s=13, marker="s", color="black", edgecolor="black", linewidth=0.65, zorder=8, label=None)


def draw_2d_regions(ax) -> None:
    az = np.linspace(AZ_MIN, AZ_MAX, 721)
    el = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 361)
    az_grid, el_grid = np.meshgrid(az, el)
    vectors = unit_vectors_from_az_el(az_grid, el_grid)

    phase2_mask = (vectors @ normalize(PHASE2_CENTER)) >= np.cos(np.deg2rad(PHASE2_HALF_ANGLE_DEG))
    phase4_mask = (vectors @ normalize(PHASE4_CENTER)) >= np.cos(np.deg2rad(PHASE4_HALF_ANGLE_DEG))
    sun_mask = (vectors @ normalize(SUN_DIRECTION)) >= np.cos(np.deg2rad(SUN_HALF_ANGLE_DEG))

    ax.contourf(az_grid, el_grid, phase2_mask, levels=[0.5, 1.5], colors=["#0072B2"], alpha=0.13)
    ax.contour(az_grid, el_grid, phase2_mask, levels=[0.5], colors=["#0072B2"], linewidths=0.75)
    ax.contourf(az_grid, el_grid, phase4_mask, levels=[0.5, 1.5], colors=["#D55E00"], alpha=0.13)
    ax.contour(az_grid, el_grid, phase4_mask, levels=[0.5], colors=["#D55E00"], linewidths=0.75)
    ax.contourf(az_grid, el_grid, sun_mask, levels=[0.5, 1.5], colors=["#CC79A7"], alpha=0.17)
    ax.contour(az_grid, el_grid, sun_mask, levels=[0.5], colors=["#CC79A7"], linewidths=0.75)

    ax.text(-0.5 * np.pi, -1.20, "P2", color="#005A8B", fontsize=6.6, weight="bold", ha="center", va="center")
    ax.text(-1.9, 0.36, "P4", color="#A54200", fontsize=6.6, weight="bold", ha="center", va="center")
    ax.text(0.0, 0.0, "Sun", color="#8F3F75", fontsize=6.6, weight="bold", ha="center", va="center")


def make_2d_plot(cases: list[AttitudeCase], output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    draw_2d_regions(ax)
    for case in cases:
        plot_trajectory_2d(ax, case, solver_label=case.name if len(cases) > 1 else None)

    ax.set_xlim(AZ_MIN, AZ_MAX)
    ax.set_ylim(-0.5 * np.pi, 0.5 * np.pi)
    ax.set_xlabel("Azimuth (rad)", labelpad=1.5)
    ax.set_ylabel("Elevation (rad)", labelpad=1.5)
    ax.grid(True, color="0.86", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xticks([AZ_MIN, -np.pi, -0.5 * np.pi, 0.0, AZ_MAX])
    ax.set_xticklabels([r"$-3\pi/2$", r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$"])
    ax.set_yticks([-0.5 * np.pi, -1.0, 0.0, 1.0, 0.5 * np.pi])
    ax.set_yticklabels([r"$-\pi/2$", "-1", "0", "1", r"$\pi/2$"])

    add_compact_legend(ax, cases, loc="upper left", bbox_to_anchor=(0.012, 0.985), include_regions=False, ncol=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_3d(ax, case: AttitudeCase, solver_label: str | None = None) -> None:
    for phase in sorted(PHASE_STYLES):
        mask = case.phases == phase
        if np.count_nonzero(mask) < 2:
            continue
        points = case.boresight[mask]
        phase_style = PHASE_STYLES[phase]
        solver_style = SOLVER_STYLES[case.name]
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=phase_style["color"],
            linestyle=solver_style["linestyle"],
            linewidth=solver_style["linewidth"],
            alpha=solver_style["alpha"],
            zorder=solver_style["zorder"],
        )
    ax.scatter(*case.boresight[0], s=34, marker="o", color="white", edgecolor="black", depthshade=False)
    ax.scatter(*case.boresight[-1], s=38, marker="s", color="black", edgecolor="black", depthshade=False)


def make_3d_plot(cases: list[AttitudeCase], output: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(7.4, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    sphere = unit_sphere()
    ax.plot_surface(*sphere, alpha=0.05, color="0.86", linewidth=0)
    ax.plot_wireframe(*sphere, alpha=0.24, color="0.48", linewidth=0.45, rstride=4, cstride=4)

    p2_cap = spherical_cap(PHASE2_CENTER, PHASE2_HALF_ANGLE_DEG)
    p4_cap = spherical_cap(PHASE4_CENTER, PHASE4_HALF_ANGLE_DEG)
    sun_cap = spherical_cap(SUN_DIRECTION, SUN_HALF_ANGLE_DEG)
    p2_cone = cone_side(PHASE2_CENTER, PHASE2_HALF_ANGLE_DEG)
    p4_cone = cone_side(PHASE4_CENTER, PHASE4_HALF_ANGLE_DEG)
    sun_cone = cone_side(SUN_DIRECTION, SUN_HALF_ANGLE_DEG)
    ax.plot_surface(*p2_cone, color="#0072B2", alpha=0.08, linewidth=0)
    ax.plot_surface(*p4_cone, color="#D55E00", alpha=0.08, linewidth=0)
    ax.plot_surface(*sun_cone, color="#CC79A7", alpha=0.08, linewidth=0)
    ax.plot_surface(*p2_cap, color="#0072B2", alpha=0.24, linewidth=0)
    ax.plot_surface(*p4_cap, color="#D55E00", alpha=0.24, linewidth=0)
    ax.plot_surface(*sun_cap, color="#CC79A7", alpha=0.26, linewidth=0)

    for case in cases:
        plot_trajectory_3d(ax, case, solver_label=case.name if len(cases) > 1 else None)

    for direction, color, label in [
        (PHASE2_CENTER, "#0072B2", "P2"),
        (PHASE4_CENTER, "#D55E00", "P4"),
        (SUN_DIRECTION, "#CC79A7", "Sun"),
    ]:
        d = normalize(direction)
        ax.quiver(0, 0, 0, d[0], d[1], d[2], color=color, linewidth=1.4, arrow_length_ratio=0.08)
        ax.text(*(1.08 * d), label, color=color, fontsize=10, weight="bold")

    ax.set_title("Unit-Sphere Constraint Geometry")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.view_init(elev=23, azim=-55)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))

    add_compact_legend(ax, cases, loc="upper left", bbox_to_anchor=(-0.02, 0.98), include_regions=True, ncol=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_compact_legend(
    ax,
    cases: list[AttitudeCase],
    loc: str,
    bbox_to_anchor: tuple[float, float],
    include_regions: bool = True,
    ncol: int = 1,
) -> None:
    region_handles = [
        mpatches.Patch(facecolor="#0072B2", alpha=0.24, edgecolor="#0072B2", label=f"P2 feasible ({PHASE2_HALF_ANGLE_DEG:.0f} deg)"),
        mpatches.Patch(facecolor="#D55E00", alpha=0.24, edgecolor="#D55E00", label=f"P4 feasible ({PHASE4_HALF_ANGLE_DEG:.0f} deg)"),
        mpatches.Patch(facecolor="#CC79A7", alpha=0.26, edgecolor="#CC79A7", label=f"Sun exclusion ({SUN_HALF_ANGLE_DEG:.0f} deg)"),
    ]
    phase_handles = [
        mlines.Line2D([], [], color=style["color"], linewidth=1.6, label=style["label"].replace("Phase ", "P"))
        for _, style in sorted(PHASE_STYLES.items())
    ]
    if len(cases) > 1:
        solver_handles = [
            mlines.Line2D(
                [],
                [],
                color="0.15",
                linestyle=(0, (1.4, 1.0)) if case.name == "Rockit/IPOPT" else SOLVER_STYLES[case.name]["linestyle"],
                linewidth=1.8 if case.name == "Rockit/IPOPT" else 1.6,
                label=case.name,
            )
            for case in cases
        ]
    else:
        legend_linestyle = (
            (0, (1.4, 1.0))
            if cases[0].name == "Rockit/IPOPT"
            else SOLVER_STYLES[cases[0].name]["linestyle"]
        )
        solver_handles = [
            mlines.Line2D(
                [],
                [],
                color="0.15",
                linestyle=legend_linestyle,
                linewidth=1.8 if cases[0].name == "Rockit/IPOPT" else 1.6,
                label=cases[0].name,
            )
        ]
    handles = (region_handles if include_regions else []) + phase_handles + solver_handles
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=6.0,
        frameon=True,
        framealpha=0.88,
        borderpad=0.18,
        handlelength=1.9,
        handletextpad=0.35,
        labelspacing=0.14,
        columnspacing=0.6,
        ncol=ncol,
    )


def make_all_plots(ddp_state_csv: Path, rockit_csv: Path, output_dir: Path, dpi: int) -> list[Path]:
    ddp = load_ddp(ddp_state_csv)
    rockit = load_rockit(rockit_csv)
    outputs = [
        output_dir / "rockit_attitude_2d.png",
        output_dir / "rockit_attitude_3d.png",
        output_dir / "ddp_attitude_2d.png",
        output_dir / "ddp_attitude_3d.png",
        output_dir / "overlay_attitude_2d.png",
        output_dir / "overlay_attitude_3d.png",
    ]
    make_2d_plot([rockit], outputs[0], dpi)
    make_3d_plot([rockit], outputs[1], dpi)
    make_2d_plot([ddp], outputs[2], dpi)
    make_3d_plot([ddp], outputs[3], dpi)
    make_2d_plot([ddp, rockit], outputs[4], dpi)
    make_3d_plot([ddp, rockit], outputs[5], dpi)
    return outputs


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent / "core data" / "attitude"
    parser = argparse.ArgumentParser(description="Plot attitude constraint geometry and boresight trajectories.")
    parser.add_argument("--ddp-state-csv", type=Path, default=base / "4phase_attitude_state_ver1.csv")
    parser.add_argument("--ddp-control-csv", type=Path, default=base / "4phase_attitude_control_ver1.csv")
    parser.add_argument("--rockit-csv", type=Path, default=base / "rotational_solution.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures" / "attitude_constraint_geometry")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The DDP control CSV is accepted to keep the CLI tied to the full DDP data set,
    # although these boresight plots only need attitude states.
    if not args.ddp_control_csv.exists():
        raise FileNotFoundError(args.ddp_control_csv)
    outputs = make_all_plots(args.ddp_state_csv, args.rockit_csv, args.output_dir, args.dpi)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
