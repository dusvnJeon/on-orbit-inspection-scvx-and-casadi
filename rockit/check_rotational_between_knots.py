from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from problem_parameters import PARAMS


@dataclass(frozen=True)
class RotKnot:
    phase: int
    node: int
    time: float
    x: np.ndarray
    tau: np.ndarray


def _optional_float(value: str) -> float:
    return float(value) if value != "" else float("nan")


def load_rotational_solution(csv_path: str | Path) -> dict[int, list[RotKnot]]:
    phases: dict[int, list[RotKnot]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase = int(row["phase"])
            knot = RotKnot(
                phase=phase,
                node=int(row["node"]),
                time=float(row["time"]),
                x=np.array(
                    [
                        float(row["qw"]),
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                        float(row["wx"]),
                        float(row["wy"]),
                        float(row["wz"]),
                    ],
                    dtype=float,
                ),
                tau=np.array(
                    [
                        _optional_float(row["taux"]),
                        _optional_float(row["tauy"]),
                        _optional_float(row["tauz"]),
                    ],
                    dtype=float,
                ),
            )
            phases.setdefault(phase, []).append(knot)

    for knots in phases.values():
        knots.sort(key=lambda item: item.node)
    return dict(sorted(phases.items()))


def quat_mul(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.array(
        [
            qw * pw - qx * px - qy * py - qz * pz,
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
        ],
        dtype=float,
    )


def quat_exp(delta_theta: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(delta_theta))
    if angle < 1e-14:
        return np.array([1.0, 0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2]], dtype=float)
    half = 0.5 * angle
    return np.r_[np.cos(half), np.sin(half) * delta_theta / angle]


def attitude_step_lie_numpy(xi: np.ndarray, tau: np.ndarray, h: float, inertia_diag: np.ndarray) -> np.ndarray:
    q = xi[:4]
    w = xi[4:7]
    j = np.asarray(inertia_diag, dtype=float)
    j_w = j * w
    w_dot = (tau - np.cross(w, j_w)) / j
    w_next = w + h * w_dot
    q_next = quat_mul(q, quat_exp(h * w))
    q_next = q_next / np.linalg.norm(q_next)
    return np.r_[q_next, w_next]


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


def constraint_values(
    xi: np.ndarray,
    tau: np.ndarray,
    phase: int,
    boresight_body: np.ndarray,
) -> dict[str, float]:
    camera_axis = q_to_rotation_matrix(xi[:4]) @ boresight_body
    omega = xi[4:7]

    values = {
        "omega_abs_margin": float(np.max(np.abs(omega)) - PARAMS.omega_max),
        "tau_abs_margin": float(np.max(np.abs(tau)) - PARAMS.tau_max),
        "q_norm_error": float(abs(np.linalg.norm(xi[:4]) - 1.0)),
        "sun_margin": float(np.dot(PARAMS.c_w3_sun, camera_axis) - np.cos(PARAMS.alpha)),
    }

    if phase == 2:
        values["phase2_los_margin"] = float(
            np.cos(PARAMS.alpha - PARAMS.theta_los) + np.dot(PARAMS.c_n_los, camera_axis)
        )
    else:
        values["phase2_los_margin"] = float("-inf")

    if phase == 4:
        values["phase4_dock_margin"] = float(
            np.cos(PARAMS.alpha - PARAMS.theta_dock) + np.dot(PARAMS.c_n_dock, camera_axis)
        )
    else:
        values["phase4_dock_margin"] = float("-inf")

    return values


def _update_worst(worst: dict[str, dict], name: str, margin: float, phase: int, node: int, time: float) -> None:
    if name not in worst or margin > worst[name]["margin"]:
        worst[name] = {"margin": margin, "phase": phase, "node": node, "time": time}


def inspect_between_knots(
    phases: dict[int, list[RotKnot]],
    substep: float,
    boresight_body: np.ndarray,
    qnorm_tol: float = 1e-6,
    output_samples_csv: str | Path | None = None,
) -> dict[str, object]:
    worst: dict[str, dict] = {}
    max_dynamics_residual = 0.0
    sample_rows = []

    for phase, knots in phases.items():
        for knot, nxt in zip(knots[:-1], knots[1:]):
            if not np.all(np.isfinite(knot.tau)):
                continue
            interval = nxt.time - knot.time
            if interval <= 0.0:
                continue

            coarse_next = attitude_step_lie_numpy(knot.x, knot.tau, interval, PARAMS.inertia_diag)
            residual = float(np.linalg.norm(coarse_next - nxt.x, ord=np.inf))
            max_dynamics_residual = max(max_dynamics_residual, residual)

            n_sub = max(1, int(np.ceil(interval / substep)))
            h_sub = interval / n_sub
            xi = knot.x.copy()

            for sub_index in range(n_sub + 1):
                t = knot.time + sub_index * h_sub
                vals = constraint_values(xi, knot.tau, phase, boresight_body)
                for name, margin in vals.items():
                    _update_worst(worst, name, margin, phase, knot.node, t)

                if output_samples_csv is not None:
                    sample_rows.append(
                        {
                            "phase": phase,
                            "interval_node": knot.node,
                            "sub_index": sub_index,
                            "time": t,
                            "qw": xi[0],
                            "qx": xi[1],
                            "qy": xi[2],
                            "qz": xi[3],
                            "wx": xi[4],
                            "wy": xi[5],
                            "wz": xi[6],
                            **vals,
                        }
                    )

                if sub_index < n_sub:
                    xi = attitude_step_lie_numpy(xi, knot.tau, h_sub, PARAMS.inertia_diag)

    if output_samples_csv is not None:
        output_path = Path(output_samples_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)

    return {"worst": worst, "max_dynamics_residual": max_dynamics_residual, "qnorm_tol": qnorm_tol}


def boresight_from_name(name: str) -> np.ndarray:
    options = {
        "x": np.array([1.0, 0.0, 0.0], dtype=float),
        "y": np.array([0.0, 1.0, 0.0], dtype=float),
        "z": np.array([0.0, 0.0, 1.0], dtype=float),
    }
    return options[name]


def print_report(result: dict[str, object]) -> None:
    print("\nBetween-knot rotational constraint check")
    print(f"  max rollout residual vs saved next knot: {result['max_dynamics_residual']:.6e}")
    print("\nWorst margins; positive means violation.")
    for name, item in sorted(result["worst"].items()):
        margin = item["margin"]
        if not np.isfinite(margin):
            continue
        if name == "q_norm_error":
            status = "VIOLATION" if margin > result["qnorm_tol"] else "ok"
        else:
            status = "VIOLATION" if margin > 0.0 else "ok"
        print(
            f"  {name:<22} {margin: .6e}  {status:<9} "
            f"phase={item['phase']} interval_node={item['node']} time={item['time']:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check rotational constraints between coarse Rockit knots.")
    parser.add_argument("--csv", type=Path, default=Path("rockit_outputs") / "rotational_solution.csv")
    parser.add_argument("--substep", type=float, default=0.05, help="Dense validation step in seconds.")
    parser.add_argument("--qnorm-tol", type=float, default=1e-6)
    parser.add_argument(
        "--boresight-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Body-frame sensor boresight axis used by rockit_implement_2.py.",
    )
    parser.add_argument("--samples-csv", type=Path, default=None, help="Optional dense sample dump CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phases = load_rotational_solution(args.csv)
    result = inspect_between_knots(
        phases,
        substep=args.substep,
        boresight_body=boresight_from_name(args.boresight_axis),
        qnorm_tol=args.qnorm_tol,
        output_samples_csv=args.samples_csv,
    )
    print_report(result)


if __name__ == "__main__":
    main()
