import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamics_residuals import load_alipddp_translation_history
from problem_parameters import PARAMS as p


FIGURE7_ELEV = 22
# FIGURE7_AZIM = -135
FIGURE7_AZIM = 20


def parse_optional_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def resolve_input_path(csv_path):
    path = Path(csv_path)
    if path.exists() or path.is_absolute():
        return path

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / path,
        script_dir.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def read_solution_csv(csv_path):
    rows = []
    with resolve_input_path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "phase": int(row["phase"]),
                    "node": int(row["node"]),
                    "time": float(row["time"]),
                    "tau": float(row["tau"]),
                    "T_i": float(row["T_i"]),
                    "px": float(row["px"]),
                    "py": float(row["py"]),
                    "pz": float(row["pz"]),
                    "vx": float(row["vx"]),
                    "vy": float(row["vy"]),
                    "vz": float(row["vz"]),
                    "Fx": parse_optional_float(row["Fx"]),
                    "Fy": parse_optional_float(row["Fy"]),
                    "Fz": parse_optional_float(row["Fz"]),
                }
            )
    return rows


def rows_from_phase_knots(phases):
    rows = []
    for phase in sorted(phases):
        knots = phases[phase]
        n_intervals = len(knots) - 1
        for knot in knots:
            tau = knot.node / n_intervals if n_intervals > 0 else 0.0
            rows.append(
                {
                    "phase": knot.phase,
                    "node": knot.node,
                    "time": knot.time,
                    "tau": tau,
                    "T_i": knot.T_i,
                    "px": knot.x[0],
                    "py": knot.x[1],
                    "pz": knot.x[2],
                    "vx": knot.x[3],
                    "vy": knot.x[4],
                    "vz": knot.x[5],
                    "Fx": knot.u[0],
                    "Fy": knot.u[1],
                    "Fz": knot.u[2],
                }
            )
    return rows


def plot_ellipsoid(ax, center, axes, color, alpha):
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    x = center[0] + axes[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + axes[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + axes[2] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def orthonormal_basis(axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return axis, e1, e2


def plot_cone(ax, apex, axis, half_angle, length, color, alpha):
    axis, e1, e2 = orthonormal_basis(axis)
    s = np.linspace(0.0, length, 30)
    phi = np.linspace(0.0, 2 * np.pi, 48)
    S, Phi = np.meshgrid(s, phi)
    R = S * np.tan(half_angle)
    points = (
        np.asarray(apex, dtype=float)
        + S[..., None] * axis
        + R[..., None] * (np.cos(Phi)[..., None] * e1 + np.sin(Phi)[..., None] * e2)
    )
    ax.plot_surface(points[..., 0], points[..., 1], points[..., 2], color=color, alpha=alpha, linewidth=0)


def set_equal_3d_axes(ax, points):
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * np.max(maxs - mins), 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_trajectory_figure7(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:pink"]
    all_points = []
    for phase_num, color in enumerate(colors, start=1):
        phase_rows = [row for row in rows if row["phase"] == phase_num]
        if not phase_rows:
            continue
        pts = np.array([[row["px"], row["py"], row["pz"]] for row in phase_rows], dtype=float)
        all_points.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.0, label=f"Phase {phase_num}")

    if not all_points:
        raise ValueError("No trajectory points found in CSV.")

    all_points = np.vstack(all_points)

    plot_ellipsoid(ax, p.c_t_c, p.e_o, "lightcoral", 0.22)
    plot_ellipsoid(ax, p.c_t_c, p.e_i, "khaki", 0.13)
    plot_cone(ax, p.c_t_los, p.c_n_los, p.theta_los, 100.0, "lightblue", 0.28)
    plot_cone(ax, p.c_t_dock, p.c_n_dock, p.theta_dock, 80.0, "seagreen", 0.28)

    ax.scatter(*p.p_init, s=80, color="tab:blue", marker="o", label="Initial Position")
    ax.scatter(*p.p_final, s=90, color="tab:red", marker="P", label="Docking Position")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D results of optimized position trajectory")
    set_equal_3d_axes(ax, np.vstack([all_points, p.c_t_c + p.e_i, p.c_t_c - p.e_i]))
    ax.view_init(elev=FIGURE7_ELEV, azim=FIGURE7_AZIM)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    except PermissionError:
        output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot translational trajectory from a saved Rockit CSV solution.")
    parser.add_argument(
        "--csv",
        default="./rockit_outputs/scvx_translation_solution.csv",
        help="Path to saved solution CSV.",
    )
    parser.add_argument(
        "--state-history-csv",
        type=Path,
        default=None,
        help="Path to ALIPDDP *_state_history.csv. If set, control/meta paths are inferred unless provided.",
    )
    parser.add_argument("--control-history-csv", type=Path, default=None)
    parser.add_argument("--meta-history-csv", type=Path, default=None)
    parser.add_argument("--outer-iter", type=int, default=None)
    parser.add_argument("--inner-iter", type=int, default=None)
    parser.add_argument(
        "--output",
        default="./rockit_outputs/scvx_figure7_trajectory_from_csv.png",
        help="Path to save the trajectory figure.",
    )
    args = parser.parse_args()

    if args.state_history_csv is not None:
        phases = load_alipddp_translation_history(
            args.state_history_csv,
            control_csv_path=args.control_history_csv,
            meta_csv_path=args.meta_history_csv,
            outer_iter=args.outer_iter,
            inner_iter=args.inner_iter,
        )
        rows = rows_from_phase_knots(phases)
    else:
        rows = read_solution_csv(args.csv)
    output_path = plot_trajectory_figure7(rows, args.output)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
