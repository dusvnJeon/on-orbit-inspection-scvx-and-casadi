from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OMEGA_MAX = 0.3
TAU_MAX = 0.016
COLORS = ("red", "blue", "green")
LINESTYLES = ("-", "--", "-.", ":")


def phase_start_times(df: pd.DataFrame) -> list[float]:
    starts = []
    seen = set()
    for _, row in df.sort_values("time").iterrows():
        phase = int(row["phase"])
        if phase not in seen:
            seen.add(phase)
            starts.append(float(row["time"]))
    return starts


def plot_components_by_phase(
    ax,
    df: pd.DataFrame,
    columns: tuple[str, str, str],
    labels: tuple[str, str, str],
    ylabel: str,
    bound: float,
) -> None:
    for phase_index, phase in enumerate(sorted(df["phase"].dropna().astype(int).unique())):
        phase_rows = df[df["phase"].astype(int) == phase].sort_values("time")
        time = phase_rows["time"].to_numpy(dtype=float)
        linestyle = LINESTYLES[phase_index % len(LINESTYLES)]

        for component_index, column in enumerate(columns):
            values = phase_rows[column].to_numpy(dtype=float)
            valid = np.isfinite(time) & np.isfinite(values)
            label = labels[component_index] if phase_index == 0 else None
            ax.plot(
                time[valid],
                values[valid],
                color=COLORS[component_index],
                linestyle=linestyle,
                linewidth=2.4,
                label=label,
            )

    ax.axhline(bound, color="0.3", linestyle=(0, (1.5, 2.5)), linewidth=1.6)
    ax.axhline(-bound, color="0.3", linestyle=(0, (1.5, 2.5)), linewidth=1.6)
    for switch_time in phase_start_times(df)[1:]:
        ax.axvline(switch_time, color="0.45", linewidth=2.2, alpha=0.9)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color="0.75", linewidth=0.8, alpha=0.45)
    ax.minorticks_on()
    ax.legend(loc="lower right", frameon=False, fontsize=11)


def plot_omega_tau(state_df: pd.DataFrame, control_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)

    plot_components_by_phase(
        axes[0],
        control_df,
        columns=("taux", "tauy", "tauz"),
        labels=(r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"),
        ylabel=r"$\tau$ (N m)",
        bound=TAU_MAX,
    )
    plot_components_by_phase(
        axes[1],
        state_df,
        columns=("wx", "wy", "wz"),
        labels=(r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"),
        ylabel=r"$\omega$ (rad/s)",
        bound=OMEGA_MAX,
    )

    t_max = max(float(state_df["time"].max()), float(control_df["time"].max()))
    axes[1].set_xlabel(r"$Time$ (s)", fontsize=13)
    axes[0].set_xlim(left=0.0, right=t_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_rockit_rotational(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    state_cols = ["phase", "node", "time", "wx", "wy", "wz"]
    control_cols = ["phase", "node", "time", "taux", "tauy", "tauz"]
    missing = set(state_cols + control_cols).difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    state_df = df[state_cols].copy()
    control_df = df[control_cols].copy()
    return state_df, control_df


def load_split_attitude(state_csv: Path, control_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_df = pd.read_csv(state_csv)
    control_df = pd.read_csv(control_csv)

    state_required = {"time", "wx", "wy", "wz", "phase"}
    control_required = {"time", "u0", "u1", "u2", "phase"}
    missing_state = state_required.difference(state_df.columns)
    missing_control = control_required.difference(control_df.columns)
    if missing_state:
        raise ValueError(f"{state_csv} is missing columns: {sorted(missing_state)}")
    if missing_control:
        raise ValueError(f"{control_csv} is missing columns: {sorted(missing_control)}")

    control_df = control_df.rename(columns={"u0": "taux", "u1": "tauy", "u2": "tauz"})
    state_df = state_df[["phase", "time", "wx", "wy", "wz"]].copy()
    control_df = control_df[["phase", "time", "taux", "tauy", "tauz"]].copy()
    return state_df, control_df


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent / "attitude"
    parser = argparse.ArgumentParser(description="Plot attitude torque/angular-velocity histories.")
    parser.add_argument("--attitude-dir", type=Path, default=base)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-rockit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    made = []

    if not args.skip_split:
        state_csv = args.attitude_dir / "4phase_attitude_state_ver1.csv"
        control_csv = args.attitude_dir / "4phase_attitude_control_ver1.csv"
        if state_csv.exists() and control_csv.exists():
            state_df, control_df = load_split_attitude(state_csv, control_csv)
            made.append(plot_omega_tau(state_df, control_df, args.output_dir / "figure9_omega_tau_attitude_ver1.png"))

    if not args.skip_rockit:
        rockit_csv = args.attitude_dir / "rotational_solution.csv"
        if rockit_csv.exists():
            state_df, control_df = load_rockit_rotational(rockit_csv)
            made.append(plot_omega_tau(state_df, control_df, args.output_dir / "figure9_omega_tau_rockit_attitude.png"))

    if not made:
        raise FileNotFoundError(f"No supported attitude CSVs found in {args.attitude_dir}.")
    for path in made:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
