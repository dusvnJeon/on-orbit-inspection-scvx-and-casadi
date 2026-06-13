from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OMEGA_MAX = 0.3
TAU_MAX = 0.016
BORESIGHT_BODY = np.array([0.0, 1.0, 0.0], dtype=float)
EXCLUSION_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=float)
PHASE2_TARGET = np.array([0.0, 0.0, -1.0], dtype=float)
PHASE4_TARGET = np.array([0.0, -1.0, 0.0], dtype=float)
EXCLUSION_MIN_ANGLE_DEG = 60.0
PHASE2_MAX_ANGLE_DEG = 20.0
PHASE4_MAX_ANGLE_DEG = 45.0
ALPHA_DEG = 60.0
THETA_LOS_DEG = 20.0
THETA_DOCK_DEG = 45.0


def q_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=float,
    )


def angle_deg(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    dots = np.clip(points @ direction, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def add_geometry_columns(state_df: pd.DataFrame) -> pd.DataFrame:
    state_df = state_df.copy()
    quats = state_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    boresight = np.array([q_to_rotation_matrix(q) @ BORESIGHT_BODY for q in quats])
    state_df[["boresight_x", "boresight_y", "boresight_z"]] = boresight
    state_df["angle_exclusion_deg"] = angle_deg(boresight, EXCLUSION_DIRECTION)
    state_df["angle_phase2_deg"] = angle_deg(boresight, PHASE2_TARGET)
    state_df["angle_phase4_deg"] = angle_deg(boresight, PHASE4_TARGET)
    return state_df


def load_split_attitude(state_csv: Path, control_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = pd.read_csv(state_csv)
    control = pd.read_csv(control_csv).rename(columns={"u0": "taux", "u1": "tauy", "u2": "tauz"})

    state_required = {"time", "wx", "wy", "wz", "qw", "qx", "qy", "qz", "phase"}
    control_required = {"time", "taux", "tauy", "tauz", "phase"}
    missing_state = state_required.difference(state.columns)
    missing_control = control_required.difference(control.columns)
    if missing_state:
        raise ValueError(f"{state_csv} is missing columns: {sorted(missing_state)}")
    if missing_control:
        raise ValueError(f"{control_csv} is missing columns: {sorted(missing_control)}")

    return add_geometry_columns(state), control


def load_rockit_attitude(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {"phase", "node", "time", "qw", "qx", "qy", "qz", "wx", "wy", "wz", "taux", "tauy", "tauz"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    state = add_geometry_columns(df.copy())
    control = df[["phase", "node", "time", "taux", "tauy", "tauz"]].copy()
    return state, control


def control_weights(control: pd.DataFrame) -> np.ndarray:
    control = control.sort_values(["phase", "time"]).copy()
    weights = np.zeros(len(control), dtype=float)

    offset = 0
    for _, phase_rows in control.groupby("phase", sort=True):
        phase_rows = phase_rows.sort_values("time")
        idx = phase_rows.index.to_numpy()
        times = phase_rows["time"].to_numpy(dtype=float)

        use_mask = np.ones(len(phase_rows), dtype=bool)
        if "node" in phase_rows.columns:
            use_mask = phase_rows["node"].to_numpy(dtype=float) < float(np.nanmax(phase_rows["node"]))

        positive_diffs = np.diff(times)
        positive_diffs = positive_diffs[positive_diffs > 0.0]
        nominal_dt = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0

        local_weights = np.zeros(len(phase_rows), dtype=float)
        for k in range(len(phase_rows)):
            if not use_mask[k]:
                continue
            if k + 1 < len(phase_rows) and times[k + 1] > times[k]:
                local_weights[k] = times[k + 1] - times[k]
            else:
                local_weights[k] = nominal_dt
        weights[offset : offset + len(phase_rows)] = local_weights
        offset += len(phase_rows)

    return weights


def compute_costs(control: pd.DataFrame) -> dict[str, float]:
    control = control.sort_values(["phase", "time"]).copy().reset_index(drop=True)
    weights = control_weights(control)
    tau = control[["taux", "tauy", "tauz"]].to_numpy(dtype=float)
    finite = np.isfinite(tau).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    tau_norm = np.linalg.norm(tau[finite], axis=1)
    return {
        "l2_tau_cost": float(np.sum(weights[finite] * tau_norm)),
        "squared_tau_cost": float(np.sum(weights[finite] * tau_norm**2)),
        "max_abs_tau": float(np.nanmax(np.abs(tau))),
    }


def compute_phase_costs(control: pd.DataFrame) -> dict[int, dict[str, float]]:
    control = control.sort_values(["phase", "time"]).copy().reset_index(drop=True)
    weights = control_weights(control)
    tau = control[["taux", "tauy", "tauz"]].to_numpy(dtype=float)
    tau_norm = np.linalg.norm(tau, axis=1)
    phases = control["phase"].astype(int).to_numpy()

    phase_costs = {}
    for phase in sorted(np.unique(phases)):
        mask = (phases == phase) & np.isfinite(tau).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
        all_phase = phases == phase
        phase_costs[int(phase)] = {
            "l2_tau_cost": float(np.sum(weights[mask] * tau_norm[mask])),
            "squared_tau_cost": float(np.sum(weights[mask] * tau_norm[mask] ** 2)),
            "max_tau_norm": float(np.nanmax(tau_norm[all_phase])),
        }
    return phase_costs


def min_or_nan(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if len(arr) else float("nan")


def compute_margins(state: pd.DataFrame, control: pd.DataFrame) -> dict[str, float]:
    omega = state[["wx", "wy", "wz"]].to_numpy(dtype=float)
    tau = control[["taux", "tauy", "tauz"]].to_numpy(dtype=float)
    boresight = state[["boresight_x", "boresight_y", "boresight_z"]].to_numpy(dtype=float)

    phase2 = state[state["phase"].astype(int) == 2]
    phase4 = state[state["phase"].astype(int) == 4]
    phase2_boresight = phase2[["boresight_x", "boresight_y", "boresight_z"]].to_numpy(dtype=float)
    phase4_boresight = phase4[["boresight_x", "boresight_y", "boresight_z"]].to_numpy(dtype=float)

    alpha = np.deg2rad(ALPHA_DEG)
    theta_los = np.deg2rad(THETA_LOS_DEG)
    theta_dock = np.deg2rad(THETA_DOCK_DEG)
    c_n_los = np.array([0.0, 0.0, 1.0], dtype=float)
    c_n_dock = np.array([0.0, 1.0, 0.0], dtype=float)

    return {
        "omega_component_margin": OMEGA_MAX - float(np.nanmax(np.abs(omega))),
        "tau_component_margin": TAU_MAX - float(np.nanmax(np.abs(tau))),
        "sun_constraint_margin": float(np.nanmin(np.cos(alpha) - boresight @ EXCLUSION_DIRECTION)),
        "phase2_constraint_margin": float(np.nanmin(-(phase2_boresight @ c_n_los) - np.cos(alpha - theta_los))),
        "phase4_constraint_margin": float(np.nanmin(-(phase4_boresight @ c_n_dock) - np.cos(alpha - theta_dock))),
        "exclusion_angle_margin_deg": min_or_nan(state["angle_exclusion_deg"] - EXCLUSION_MIN_ANGLE_DEG),
        "phase2_rockit_angle_margin_deg": min_or_nan(np.degrees(alpha - theta_los) - phase2["angle_phase2_deg"]),
        "phase4_rockit_angle_margin_deg": min_or_nan(np.degrees(alpha - theta_dock) - phase4["angle_phase4_deg"]),
        "phase2_angle_margin_deg": min_or_nan(PHASE2_MAX_ANGLE_DEG - phase2["angle_phase2_deg"]),
        "phase4_angle_margin_deg": min_or_nan(PHASE4_MAX_ANGLE_DEG - phase4["angle_phase4_deg"]),
        "q_norm_error_max": float(
            np.nanmax(np.abs(np.linalg.norm(state[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float), axis=1) - 1.0))
        ),
    }


def summarize_case(name: str, state: pd.DataFrame, control: pd.DataFrame) -> dict[str, float | str]:
    state = add_geometry_columns(state)
    costs = compute_costs(control)
    phase_costs = compute_phase_costs(control)
    margins = compute_margins(state, control)
    return {"case": name, **costs, "phase_costs": phase_costs, **margins}


def print_summary(rows: list[dict[str, float | str]]) -> None:
    def fmt(value: float, unit: str = "") -> str:
        if not np.isfinite(value):
            return "n/a"
        abs_value = abs(value)
        if 1.0e-3 <= abs_value < 1.0e4:
            text = f"{value:.6f}"
        else:
            text = f"{value:.3e}"
        return f"{text}{unit}"

    def verdict(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return "OK" if value >= -1.0e-9 else "VIOLATION"

    print("Attitude CSV summary")
    print("=" * 72)
    print("Positive margins mean feasible. Negative margins mean violation.")
    print("Constraint margins use the same dot-product inequalities as rockit_implement_2.py.")

    for row in rows:
        print()
        print(f"[{row['case']}]")
        print("  Cost")
        print(f"    integral ||tau|| dt      : {fmt(float(row['l2_tau_cost']))}")
        print(f"    integral ||tau||^2 dt    : {fmt(float(row['squared_tau_cost']))}")
        print(f"    max |tau component|      : {fmt(float(row['max_abs_tau']))} / {TAU_MAX:.6f}")
        print("    phase breakdown          :")
        for phase, phase_cost in row["phase_costs"].items():
            print(
                f"      phase {phase}: "
                f"||tau|| dt={fmt(phase_cost['l2_tau_cost'])}, "
                f"||tau||^2 dt={fmt(phase_cost['squared_tau_cost'])}, "
                f"max ||tau||={fmt(phase_cost['max_tau_norm'])}"
            )

        print("  Component bounds")
        print(
            "    omega margin             : "
            f"{fmt(float(row['omega_component_margin']))}  [{verdict(float(row['omega_component_margin']))}]"
        )
        print(
            "    tau margin               : "
            f"{fmt(float(row['tau_component_margin']))}  [{verdict(float(row['tau_component_margin']))}]"
        )

        print("  Constraint margins")
        print(
            "    sun avoidance            : "
            f"{fmt(float(row['sun_constraint_margin']))}  [{verdict(float(row['sun_constraint_margin']))}]"
        )
        print(
            "    phase 2 LOS              : "
            f"{fmt(float(row['phase2_constraint_margin']))}  [{verdict(float(row['phase2_constraint_margin']))}]"
        )
        print(
            "    phase 4 docking          : "
            f"{fmt(float(row['phase4_constraint_margin']))}  [{verdict(float(row['phase4_constraint_margin']))}]"
        )

        print("  Angle margins")
        print(
            "    exclusion angle - 60 deg : "
            f"{fmt(float(row['exclusion_angle_margin_deg']), ' deg')}"
        )
        print(
            "    40 deg - phase 2 angle   : "
            f"{fmt(float(row['phase2_rockit_angle_margin_deg']), ' deg')}  [Rockit constraint]"
        )
        print(
            "    15 deg - phase 4 angle   : "
            f"{fmt(float(row['phase4_rockit_angle_margin_deg']), ' deg')}  [Rockit constraint]"
        )
        print(
            "    20 deg - phase 2 angle   : "
            f"{fmt(float(row['phase2_angle_margin_deg']), ' deg')}"
        )
        print(
            "    45 deg - phase 4 angle   : "
            f"{fmt(float(row['phase4_angle_margin_deg']), ' deg')}"
        )
        print(f"    max | ||q|| - 1 |        : {fmt(float(row['q_norm_error_max']))}")

    if len(rows) >= 2:
        print()
        print("Comparison")
        print("-" * 72)
        base = rows[0]
        for row in rows[1:]:
            print(f"{row['case']} relative to {base['case']}:")
            print(
                "  integral ||tau|| dt        : "
                f"{fmt(float(row['l2_tau_cost']) - float(base['l2_tau_cost']))}"
            )
            print(
                "  integral ||tau||^2 dt      : "
                f"{fmt(float(row['squared_tau_cost']) - float(base['squared_tau_cost']))}"
            )


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent / "attitude"
    parser = argparse.ArgumentParser(description="Summarize attitude costs and minimum constraint margins.")
    parser.add_argument("--attitude-dir", type=Path, default=base)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-rockit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    if not args.skip_split:
        split_candidates = [
            (
                args.attitude_dir / "4phase_attitude_state_ver1.csv",
                args.attitude_dir / "4phase_attitude_control_ver1.csv",
            ),
            (
                args.attitude_dir / "4phase_attitude_state_ver1_2.csv",
                args.attitude_dir / "4phase_attitude_control_ver1_2.csv",
            ),
        ]
        state_csv = control_csv = None
        for candidate_state, candidate_control in split_candidates:
            if candidate_state.exists() and candidate_control.exists():
                state_csv, control_csv = candidate_state, candidate_control
                break
        if state_csv is not None and control_csv is not None:
            state, control = load_split_attitude(state_csv, control_csv)
            rows.append(summarize_case("attitude_ver1", state, control))

    if not args.skip_rockit:
        rockit_csv = args.attitude_dir / "rotational_solution.csv"
        if rockit_csv.exists():
            state, control = load_rockit_attitude(rockit_csv)
            rows.append(summarize_case("rockit_attitude", state, control))

    if not rows:
        raise FileNotFoundError(f"No supported attitude CSVs found in {args.attitude_dir}.")
    print_summary(rows)


if __name__ == "__main__":
    main()
