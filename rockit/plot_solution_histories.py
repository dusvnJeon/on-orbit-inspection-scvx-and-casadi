from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problem_parameters import PARAMS as p


TRANSLATIONAL_COLUMNS = (
    "phase",
    "node",
    "time",
    "T_i",
    "vx",
    "vy",
    "vz",
    "Fx",
    "Fy",
    "Fz",
)
ROTATIONAL_COLUMNS = (
    "phase",
    "node",
    "time",
    "T_i",
    "wx",
    "wy",
    "wz",
    "taux",
    "tauy",
    "tauz",
)


def read_solution_rows(csv_path: str | Path, columns: tuple[str, ...]) -> list[dict[str, float | int]]:
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for column in columns:
                if column in ("phase", "node"):
                    parsed[column] = int(row[column])
                else:
                    parsed[column] = float(row[column]) if row[column] != "" else float("nan")
            rows.append(parsed)
    return rows


def read_translational_solution(csv_path: str | Path) -> list[dict[str, float | int]]:
    return read_solution_rows(csv_path, TRANSLATIONAL_COLUMNS)


def read_rotational_solution(csv_path: str | Path) -> list[dict[str, float | int]]:
    return read_solution_rows(csv_path, ROTATIONAL_COLUMNS)


def phase_start_times(rows: list[dict[str, float | int]]) -> list[float]:
    starts = []
    seen = set()
    for row in rows:
        phase = int(row["phase"])
        if phase not in seen:
            seen.add(phase)
            starts.append(float(row["time"]))
    return starts


def _plot_component_history(
    ax,
    rows: list[dict[str, float | int]],
    columns: tuple[str, str, str],
    labels: tuple[str, str, str],
    ylabel: str,
    bound: float | None = None,
    autoscale_to_data: bool = False,
) -> None:
    colors = ("red", "blue", "green")
    linestyles = ("-", "--", "-.", ":")

    phases = sorted({int(row["phase"]) for row in rows})
    for phase_index, phase in enumerate(phases):
        phase_rows = [row for row in rows if int(row["phase"]) == phase]
        time = np.array([float(row["time"]) for row in phase_rows], dtype=float)
        linestyle = linestyles[phase_index % len(linestyles)]

        for component_index, column in enumerate(columns):
            values = np.array([float(row[column]) for row in phase_rows], dtype=float)
            valid = np.isfinite(time) & np.isfinite(values)
            label = labels[component_index] if phase_index == 0 else None
            ax.plot(
                time[valid],
                values[valid],
                color=colors[component_index],
                linestyle=linestyle,
                linewidth=2.4,
                label=label,
            )

    if bound is not None:
        ax.axhline(bound, color="0.3", linestyle=(0, (1.5, 2.5)), linewidth=1.6)
        ax.axhline(-bound, color="0.3", linestyle=(0, (1.5, 2.5)), linewidth=1.6)

    for switch_time in phase_start_times(rows)[1:]:
        ax.axvline(switch_time, color="0.45", linewidth=2.2, alpha=0.9)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color="0.75", linewidth=0.8, alpha=0.45)
    ax.minorticks_on()
    ax.legend(loc="lower right", frameon=False, fontsize=11)

    if autoscale_to_data:
        values = np.array(
            [
                float(row[column])
                for row in rows
                for column in columns
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        ymin = float(np.min(values))
        ymax = float(np.max(values))
        margin = 0.08 * max(ymax - ymin, max(abs(ymin), abs(ymax)), 1e-12)
        ax.set_ylim(ymin - margin, ymax + margin)


def plot_translational_velocity_control_histories(
    rows: list[dict[str, float | int]],
    output_path: str | Path,
    show_bounds: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)

    _plot_component_history(
        axes[0],
        rows,
        columns=("Fx", "Fy", "Fz"),
        labels=(r"$u_x$", r"$u_y$", r"$u_z$"),
        ylabel=r"$u$ (N)",
        bound=p.f_max if show_bounds else None,
        autoscale_to_data=not show_bounds,
    )
    _plot_component_history(
        axes[1],
        rows,
        columns=("vx", "vy", "vz"),
        labels=(r"$v_x$", r"$v_y$", r"$v_z$"),
        ylabel=r"$v$ (m/s)",
        bound=p.v_max if show_bounds else None,
        autoscale_to_data=not show_bounds,
    )

    axes[1].set_xlabel(r"$Time$ (s)", fontsize=13)
    axes[0].set_xlim(left=0.0, right=max(float(row["time"]) for row in rows))
    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    except PermissionError:
        output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rotational_omega_tau_histories(
    rows: list[dict[str, float | int]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)

    _plot_component_history(
        axes[0],
        rows,
        columns=("taux", "tauy", "tauz"),
        labels=(r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"),
        ylabel=r"$\tau$ (N m)",
        autoscale_to_data=True,
    )
    _plot_component_history(
        axes[1],
        rows,
        columns=("wx", "wy", "wz"),
        labels=(r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"),
        ylabel=r"$\omega$ (rad/s)",
        autoscale_to_data=True,
    )

    axes[1].set_xlabel(r"$Time$ (s)", fontsize=13)
    axes[0].set_xlim(left=0.0, right=max(float(row["time"]) for row in rows))
    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    except PermissionError:
        output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved Rockit solution history figures.")
    parser.add_argument(
        "--translational-csv",
        default=Path("rockit_outputs") / "translational_solution.csv",
        type=Path,
        help="Path to translational solution CSV.",
    )
    parser.add_argument(
        "--rotational-csv",
        default=Path("rockit_outputs") / "rotational_solution.csv",
        type=Path,
        help="Path to rotational solution CSV.",
    )
    parser.add_argument(
        "--output",
        default=Path("rockit_outputs") / "figure9_velocity_control_histories.png",
        type=Path,
        help="Path to save the translational velocity/control figure.",
    )
    parser.add_argument(
        "--rotational-output",
        default=Path("rockit_outputs") / "omega_tau_histories.png",
        type=Path,
        help="Path to save the rotational omega/tau figure.",
    )
    parser.add_argument(
        "--plot",
        choices=("translational", "rotational", "both"),
        default="translational",
        help="Which history figure to save.",
    )
    parser.add_argument(
        "--no-bounds",
        action="store_true",
        help="Do not draw translational box-constraint lines; autoscale y-axes to data.",
    )
    args = parser.parse_args()

    if args.plot in ("translational", "both"):
        rows = read_translational_solution(args.translational_csv)
        output_path = plot_translational_velocity_control_histories(rows, args.output, show_bounds=not args.no_bounds)
        print(f"Saved {output_path}")

    if args.plot in ("rotational", "both"):
        rows = read_rotational_solution(args.rotational_csv)
        output_path = plot_rotational_omega_tau_histories(rows, args.rotational_output)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
