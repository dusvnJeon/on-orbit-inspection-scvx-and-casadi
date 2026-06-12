from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from problem_parameters import PARAMS


STATE_COLUMNS = ("px", "py", "pz", "vx", "vy", "vz")
CONTROL_COLUMNS = ("Fx", "Fy", "Fz")
ALIPDDP_STATE_COLUMNS = ("x", "y", "z", "vx", "vy", "vz")
ALIPDDP_CONTROL_COLUMNS = ("ux", "uy", "uz")


@dataclass(frozen=True)
class Knot:
    phase: int
    node: int
    time: float
    T_i: float
    x: np.ndarray
    u: np.ndarray


def _parse_optional_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def hcw_Ad_Bd(h: float, n: float = PARAMS.chief_mean_motion, m: float = PARAMS.mass) -> tuple[np.ndarray, np.ndarray]:
    """
    Numeric HCW exact ZOH discretization.

    x = [px, py, pz, vx, vy, vz]^T
    u = [Fx, Fy, Fz]^T
    x_next = Ad(h) @ x + Bd(h) @ u
    """
    c = np.cos(n * h)
    s = np.sin(n * h)

    Ad = np.array(
        [
            [4 - 3 * c, 0, 0, s / n, 2 * (1 - c) / n, 0],
            [6 * (s - n * h), 1, 0, -2 * (1 - c) / n, (4 * s - 3 * n * h) / n, 0],
            [0, 0, c, 0, 0, s / n],
            [3 * n * s, 0, 0, c, 2 * s, 0],
            [-6 * n * (1 - c), 0, 0, -2 * s, 4 * c - 3, 0],
            [0, 0, -n * s, 0, 0, c],
        ],
        dtype=float,
    )

    Bd = (1 / m) * np.array(
        [
            [(1 - c) / n**2, 2 * (n * h - s) / n**2, 0],
            [-2 * (n * h - s) / n**2, 4 * (1 - c) / n**2 - 1.5 * h**2, 0],
            [0, 0, (1 - c) / n**2],
            [s / n, 2 * (1 - c) / n, 0],
            [-2 * (1 - c) / n, 4 * s / n - 3 * h, 0],
            [0, 0, s / n],
        ],
        dtype=float,
    )

    return Ad, Bd


def load_translational_solution(csv_path: str | Path) -> dict[int, list[Knot]]:
    """Load translational solution CSV and return knots grouped by phase."""
    phases: dict[int, list[Knot]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phase = int(row["phase"])
            knot = Knot(
                phase=phase,
                node=int(row["node"]),
                time=float(row["time"]),
                T_i=float(row["T_i"]),
                x=np.array([float(row[col]) for col in STATE_COLUMNS], dtype=float),
                u=np.array([_parse_optional_float(row[col]) for col in CONTROL_COLUMNS], dtype=float),
            )
            phases.setdefault(phase, []).append(knot)

    for knots in phases.values():
        knots.sort(key=lambda item: item.node)
    return dict(sorted(phases.items()))


def _iteration_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["outer_iter"]), int(row["iter"])


def _select_history_rows(rows: list[dict[str, str]], target: tuple[int, int] | None) -> tuple[tuple[int, int], list[dict[str, str]]]:
    if not rows:
        raise ValueError("History CSV is empty.")
    if target is None:
        target = max(_iteration_key(row) for row in rows)
    selected = [row for row in rows if _iteration_key(row) == target]
    if not selected:
        raise ValueError(f"No rows found for outer_iter={target[0]}, iter={target[1]}.")
    return target, selected


def _infer_history_path(path: Path, old: str, new: str) -> Path:
    return path.with_name(path.name.replace(old, new))


def load_alipddp_translation_history(
    state_csv_path: str | Path,
    control_csv_path: str | Path | None = None,
    meta_csv_path: str | Path | None = None,
    outer_iter: int | None = None,
    inner_iter: int | None = None,
) -> dict[int, list[Knot]]:
    """
    Load ALIPDDP/C++ translation history CSVs and return Rockit-like phase knots.

    The expected files are:
        *_state_history.csv: outer_iter,iter,time,x,y,z,vx,vy,vz
        *_control_history.csv: outer_iter,iter,time,ux,uy,uz
        *_meta_history.csv: outer_iter,iter,N1..N4,DT1..DT4,...

    If outer_iter/inner_iter are omitted, the latest pair in the state history
    is used. The selected state rows are split as N_i + 1 rows per phase and
    control rows as N_i rows per phase.
    """
    state_csv_path = Path(state_csv_path)
    if control_csv_path is None:
        control_csv_path = _infer_history_path(state_csv_path, "_state_history.csv", "_control_history.csv")
    if meta_csv_path is None:
        meta_csv_path = _infer_history_path(state_csv_path, "_state_history.csv", "_meta_history.csv")

    with state_csv_path.open(newline="", encoding="utf-8") as f:
        state_rows_all = list(csv.DictReader(f))
    target = None if outer_iter is None and inner_iter is None else (outer_iter, inner_iter)
    if (outer_iter is None) ^ (inner_iter is None):
        raise ValueError("Specify both outer_iter and inner_iter, or neither.")
    selected_key, state_rows = _select_history_rows(state_rows_all, target)

    with Path(control_csv_path).open(newline="", encoding="utf-8") as f:
        _, control_rows = _select_history_rows(list(csv.DictReader(f)), selected_key)
    with Path(meta_csv_path).open(newline="", encoding="utf-8") as f:
        _, meta_rows = _select_history_rows(list(csv.DictReader(f)), selected_key)
    meta = meta_rows[-1]

    n_intervals_by_phase = [int(meta[f"N{i}"]) for i in range(1, 5)]
    durations = [float(meta[f"DT{i}"]) for i in range(1, 5)]
    expected_states = sum(n + 1 for n in n_intervals_by_phase)
    expected_controls = sum(n_intervals_by_phase)
    if len(state_rows) != expected_states:
        raise ValueError(f"Selected state rows={len(state_rows)} but expected {expected_states}.")
    if len(control_rows) != expected_controls:
        raise ValueError(f"Selected control rows={len(control_rows)} but expected {expected_controls}.")

    phases: dict[int, list[Knot]] = {}
    state_index = 0
    control_index = 0
    for phase_number, (n_intervals, duration) in enumerate(zip(n_intervals_by_phase, durations), start=1):
        phase_controls = control_rows[control_index : control_index + n_intervals]
        for node in range(n_intervals + 1):
            state_row = state_rows[state_index + node]
            if node < n_intervals:
                control_row = phase_controls[node]
                u = np.array([float(control_row[col]) for col in ALIPDDP_CONTROL_COLUMNS], dtype=float)
            else:
                u = np.full(3, np.nan)

            knot = Knot(
                phase=phase_number,
                node=node,
                time=float(state_row["time"]),
                T_i=duration,
                x=np.array([float(state_row[col]) for col in ALIPDDP_STATE_COLUMNS], dtype=float),
                u=u,
            )
            phases.setdefault(phase_number, []).append(knot)

        state_index += n_intervals + 1
        control_index += n_intervals

    return phases


def save_rockit_style_csv(phases: dict[int, list[Knot]], output_path: str | Path) -> Path:
    """Save any loaded phase knots using the Rockit translational CSV schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "phase",
        "node",
        "time",
        "tau",
        "T_i",
        "px",
        "py",
        "pz",
        "vx",
        "vy",
        "vz",
        "Fx",
        "Fy",
        "Fz",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for phase in sorted(phases):
            knots = phases[phase]
            n_intervals = len(knots) - 1
            for knot in knots:
                force = ["" if np.isnan(value) else value for value in knot.u]
                writer.writerow(
                    {
                        "phase": knot.phase,
                        "node": knot.node,
                        "time": knot.time,
                        "tau": knot.node / n_intervals,
                        "T_i": knot.T_i,
                        "px": knot.x[0],
                        "py": knot.x[1],
                        "pz": knot.x[2],
                        "vx": knot.x[3],
                        "vy": knot.x[4],
                        "vz": knot.x[5],
                        "Fx": force[0],
                        "Fy": force[1],
                        "Fz": force[2],
                    }
                )
    return output_path


def load_meshcat_translation_inspection(
    state_csv_path: str | Path,
    control_csv_path: str | Path | None = None,
    time_csv_path: str | Path | None = None,
) -> dict[int, list[Knot]]:
    """
    Load Meshcat translation inspection CSVs and return knots grouped by phase.

    Meshcat stores state, control, and phase timing in separate files:
        translation_inspection_state.csv: time,x,y,z,vx,vy,vz
        translation_inspection_control.csv: time,ux,uy,uz
        translation_inspection_time.csv: phase,DeltaT,N

    The control file has one value per interval, so the final knot in each phase
    reuses the previous interval's control.
    """
    state_csv_path = Path(state_csv_path)
    if control_csv_path is None:
        control_csv_path = state_csv_path.with_name(state_csv_path.name.replace("_state.csv", "_control.csv"))
    if time_csv_path is None:
        time_csv_path = state_csv_path.with_name(state_csv_path.name.replace("_state.csv", "_time.csv"))

    with state_csv_path.open(newline="", encoding="utf-8") as f:
        state_rows = list(csv.DictReader(f))
    with Path(control_csv_path).open(newline="", encoding="utf-8") as f:
        control_rows = list(csv.DictReader(f))
    with Path(time_csv_path).open(newline="", encoding="utf-8") as f:
        phase_rows = list(csv.DictReader(f))

    phases: dict[int, list[Knot]] = {}
    state_index = 0
    control_index = 0

    for phase_number, phase_row in enumerate(phase_rows, start=1):
        T_i = float(phase_row["DeltaT"])
        n_intervals = int(phase_row["N"])

        for node in range(n_intervals + 1):
            state_row = state_rows[state_index]
            control_row = control_rows[min(control_index, len(control_rows) - 1)]

            knot = Knot(
                phase=phase_number,
                node=node,
                time=float(state_row["time"]),
                T_i=T_i,
                x=np.array(
                    [
                        float(state_row["x"]),
                        float(state_row["y"]),
                        float(state_row["z"]),
                        float(state_row["vx"]),
                        float(state_row["vy"]),
                        float(state_row["vz"]),
                    ],
                    dtype=float,
                ),
                u=np.array(
                    [
                        float(control_row["ux"]),
                        float(control_row["uy"]),
                        float(control_row["uz"]),
                    ],
                    dtype=float,
                ),
            )
            phases.setdefault(phase_number, []).append(knot)

            state_index += 1
            if node < n_intervals:
                control_index += 1

        # Meshcat trajectories share the phase boundary knot instead of
        # duplicating it as the first row of the next phase.
        state_index -= 1

    return phases


def phase_step_size(knots: list[Knot]) -> float:
    """Return h = T_i / N for a phase with N intervals and N+1 knots."""
    if len(knots) < 2:
        raise ValueError("At least two knots are required to compute a phase step size.")
    return knots[0].T_i / (len(knots) - 1)


def phase_Ad_Bd(knots: list[Knot]) -> tuple[np.ndarray, np.ndarray]:
    """Compute Ad, Bd for one phase from its CSV phase duration and knot count."""
    return hcw_Ad_Bd(phase_step_size(knots))


def _find_knot(phases: dict[int, list[Knot]], phase_number: int, knot_number: int) -> Knot:
    for knot in phases[phase_number]:
        if knot.node == knot_number:
            return knot
    raise KeyError(f"phase={phase_number}, node={knot_number} was not found.")


def _rollout_residual(earlier: Knot, later: Knot, h: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    Ad, Bd = hcw_Ad_Bd(h)
    control = np.nan_to_num(earlier.u, nan=0.0)
    predicted = Ad @ earlier.x + Bd @ control
    error = later.x - predicted
    position_residual = float(np.linalg.norm(error[:3], ord=1))
    velocity_residual = float(np.linalg.norm(error[3:], ord=1))
    total_residual = position_residual + velocity_residual
    return total_residual, position_residual, velocity_residual, error, predicted


def two_knot_dynamics_residual(
    phases: dict[int, list[Knot]],
    first_phase: int,
    first_node: int,
    second_phase: int,
    second_node: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compute 1-norm dynamics residual between two indexed knots.

    The earlier knot is selected by CSV time. Its state and control are rolled out
    by h = later.time - earlier.time, then compared with the later knot state.
    Returns (
        total_residual_1_norm,
        position_residual_1_norm,
        velocity_residual_1_norm,
        error_vector,
        predicted_state,
    ).
    """
    first = _find_knot(phases, first_phase, first_node)
    second = _find_knot(phases, second_phase, second_node)

    earlier, later = (first, second) if first.time <= second.time else (second, first)
    h = later.time - earlier.time
    return _rollout_residual(earlier, later, h)


def compute_process_residuals(phases: dict[int, list[Knot]]) -> dict[str, object]:
    """
    Compute residual metrics for the full multiphase process.

    Returns:
        phase_internal_position_sums: sum of position residuals inside each phase
        phase_connection_position_residuals: position residual at phase i last knot -> phase i+1 first knot
        final_position_residual: 1-norm mismatch between final CSV position and PARAMS.p_final
        total_position_residual: sum of all position residuals, including final_position_residual

    Velocity residuals and combined state residuals are also returned for
    debugging, but _print_summary reports position residuals only.
    """
    phase_internal_position_sums: dict[int, float] = {}
    phase_internal_velocity_sums: dict[int, float] = {}
    phase_internal_sums: dict[int, float] = {}
    total_position_residual = 0.0
    total_velocity_residual = 0.0
    total_residual = 0.0

    for phase, knots in phases.items():
        phase_position_sum = 0.0
        phase_velocity_sum = 0.0
        phase_sum = 0.0
        for current, nxt in zip(knots[:-1], knots[1:]):
            residual, position_residual, velocity_residual, _, _ = _rollout_residual(
                current,
                nxt,
                nxt.time - current.time,
            )
            phase_position_sum += position_residual
            phase_velocity_sum += velocity_residual
            phase_sum += residual
        phase_internal_position_sums[phase] = phase_position_sum
        phase_internal_velocity_sums[phase] = phase_velocity_sum
        phase_internal_sums[phase] = phase_sum
        total_position_residual += phase_position_sum
        total_velocity_residual += phase_velocity_sum
        total_residual += phase_sum

    phase_connection_position_residuals: dict[tuple[int, int], float] = {}
    phase_connection_velocity_residuals: dict[tuple[int, int], float] = {}
    phase_connection_residuals: dict[tuple[int, int], float] = {}
    phase_numbers = sorted(phases)
    for current_phase, next_phase in zip(phase_numbers[:-1], phase_numbers[1:]):
        current = phases[current_phase][-1]
        nxt = phases[next_phase][0]
        residual, position_residual, velocity_residual, _, _ = _rollout_residual(
            current,
            nxt,
            nxt.time - current.time,
        )
        phase_connection_position_residuals[(current_phase, next_phase)] = position_residual
        phase_connection_velocity_residuals[(current_phase, next_phase)] = velocity_residual
        phase_connection_residuals[(current_phase, next_phase)] = residual
        total_position_residual += position_residual
        total_velocity_residual += velocity_residual
        total_residual += residual

    final_phase = phase_numbers[-1]
    final_position = phases[final_phase][-1].x[:3]
    final_position_error = final_position - PARAMS.p_final
    final_position_residual = float(np.linalg.norm(final_position_error, ord=1))
    total_position_residual += final_position_residual
    total_residual += final_position_residual

    return {
        "phase_internal_position_sums": phase_internal_position_sums,
        "phase_internal_velocity_sums": phase_internal_velocity_sums,
        "phase_internal_sums": phase_internal_sums,
        "phase_connection_position_residuals": phase_connection_position_residuals,
        "phase_connection_velocity_residuals": phase_connection_velocity_residuals,
        "phase_connection_residuals": phase_connection_residuals,
        "final_position_error": final_position_error,
        "final_position_residual": final_position_residual,
        "total_position_residual": total_position_residual,
        "total_velocity_residual": total_velocity_residual,
        "total_residual": total_residual,
    }


def compute_control_costs(phases: dict[int, list[Knot]], eps: float = 1e-6) -> dict[str, object]:
    """
    Compute control-effort costs from phase knots.

    rockit_fuel_like_cost matches the current translational Rockit objective:
        sum_i sum_k h_i * sqrt(||F_i,k||_2^2 + eps)

    squared_force_cost matches the SCvx baseline objective:
        sum_i sum_k h_i * ||F_i,k||_2^2

    Only interval controls k=0..N-1 are used; terminal-node controls are ignored.
    """
    phase_rockit_costs: dict[int, float] = {}
    phase_squared_costs: dict[int, float] = {}
    phase_l2_costs: dict[int, float] = {}

    for phase, knots in phases.items():
        h = phase_step_size(knots)
        rockit_cost = 0.0
        squared_cost = 0.0
        l2_cost = 0.0
        for knot in knots[:-1]:
            if np.any(np.isnan(knot.u)):
                raise ValueError(f"NaN control found at phase={knot.phase}, node={knot.node}.")
            u2 = float(np.dot(knot.u, knot.u))
            rockit_cost += h * np.sqrt(u2 + eps)
            squared_cost += h * u2
            l2_cost += h * np.sqrt(u2)
        phase_rockit_costs[phase] = rockit_cost
        phase_squared_costs[phase] = squared_cost
        phase_l2_costs[phase] = l2_cost

    return {
        "phase_rockit_fuel_like_costs": phase_rockit_costs,
        "phase_squared_force_costs": phase_squared_costs,
        "phase_l2_force_costs": phase_l2_costs,
        "rockit_fuel_like_cost": sum(phase_rockit_costs.values()),
        "squared_force_cost": sum(phase_squared_costs.values()),
        "l2_force_cost": sum(phase_l2_costs.values()),
    }


def _print_summary(metrics: dict[str, object]) -> None:
    print("Phase internal position residual sums:")
    for phase, value in metrics["phase_internal_position_sums"].items():
        print(f"  phase {phase}: {value:.12e}")

    print("Phase-to-phase connection position residuals:")
    for (from_phase, to_phase), value in metrics["phase_connection_position_residuals"].items():
        print(f"  phase {from_phase} -> {to_phase}: {value:.12e}")

    print(f"Final position residual: {metrics['final_position_residual']:.12e}")
    print(f"Total position residual: {metrics['total_position_residual']:.12e}")


def _print_cost_summary(costs: dict[str, object]) -> None:
    print("Phase Rockit fuel-like costs:")
    for phase, value in costs["phase_rockit_fuel_like_costs"].items():
        print(f"  phase {phase}: {value:.12e}")

    print("Phase squared-force costs:")
    for phase, value in costs["phase_squared_force_costs"].items():
        print(f"  phase {phase}: {value:.12e}")

    print(f"Rockit fuel-like cost: {costs['rockit_fuel_like_cost']:.12e}")
    print(f"L2 force cost without eps: {costs['l2_force_cost']:.12e}")
    print(f"Squared-force cost: {costs['squared_force_cost']:.12e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify HCW dynamics residuals from translational_solution.csv.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("rockit_outputs") / "translational_solution.csv",
        help="Path to translational solution CSV.",
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
        "--export-rockit-csv",
        type=Path,
        default=None,
        help="Optional path to save the selected history in Rockit translational CSV format.",
    )
    parser.add_argument("--first-phase", type=int, default=None)
    parser.add_argument("--first-node", type=int, default=None)
    parser.add_argument("--second-phase", type=int, default=None)
    parser.add_argument("--second-node", type=int, default=None)
    args = parser.parse_args()

    if args.state_history_csv is not None:
        phases = load_alipddp_translation_history(
            args.state_history_csv,
            control_csv_path=args.control_history_csv,
            meta_csv_path=args.meta_history_csv,
            outer_iter=args.outer_iter,
            inner_iter=args.inner_iter,
        )
    else:
        phases = load_translational_solution(args.csv)

    if args.export_rockit_csv is not None:
        output_path = save_rockit_style_csv(phases, args.export_rockit_csv)
        print(f"Saved Rockit-style CSV: {output_path}")

    # To use Meshcat/translation_inspection_state.csv instead, replace the
    # line above with the loader below. The control and time CSV paths are
    # inferred automatically from the state CSV filename.
    #
    # phases = load_meshcat_translation_inspection(
    #     Path("Meshcat") / "translation_inspection_state.csv"
    # )

    requested_nodes = [args.first_phase, args.first_node, args.second_phase, args.second_node]
    if any(value is not None for value in requested_nodes):
        if any(value is None for value in requested_nodes):
            parser.error("Specify all of --first-phase, --first-node, --second-phase, and --second-node.")
        residual, position_residual, velocity_residual, error, predicted = two_knot_dynamics_residual(
            phases,
            args.first_phase,
            args.first_node,
            args.second_phase,
            args.second_node,
        )
        print(f"Two-knot dynamics residual 1-norm: {residual:.12e}")
        print(f"Two-knot position residual 1-norm: {position_residual:.12e}")
        print(f"Two-knot velocity residual 1-norm: {velocity_residual:.12e}")
        print("error:", np.array2string(error, precision=12, separator=", "))
        print("predicted:", np.array2string(predicted, precision=12, separator=", "))

    _print_summary(compute_process_residuals(phases))
    _print_cost_summary(compute_control_costs(phases))


if __name__ == "__main__":
    main()
