import csv
from pathlib import Path

import numpy as np


def _as_rows(values, n_cols):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[0] == n_cols and values.shape[1] != n_cols:
        values = values.T
    return values


def _sample_stage_solution(sol, stage, x, F, T_i, phase_num, t_offset):
    tau, x_values = sol(stage).sample(x, grid="control")
    _, F_values = sol(stage).sample(F, grid="control")
    _, T_values = sol(stage).sample(T_i, grid="control")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    x_values = _as_rows(x_values, 6)
    F_values = _as_rows(F_values, 3)
    T_values = np.asarray(T_values, dtype=float).reshape(-1)

    duration = float(T_values[0])
    t = t_offset + tau * duration

    if F_values.shape[0] < x_values.shape[0]:
        pad = np.full((x_values.shape[0] - F_values.shape[0], 3), np.nan)
        F_values = np.vstack([F_values, pad])
    elif F_values.shape[0] > x_values.shape[0]:
        F_values = F_values[:x_values.shape[0], :]

    rows = []
    for k in range(x_values.shape[0]):
        rows.append(
            {
                "phase": phase_num,
                "node": k,
                "time": t[k],
                "tau": tau[k],
                "T_i": duration,
                "px": x_values[k, 0],
                "py": x_values[k, 1],
                "pz": x_values[k, 2],
                "vx": x_values[k, 3],
                "vy": x_values[k, 4],
                "vz": x_values[k, 5],
                "Fx": F_values[k, 0],
                "Fy": F_values[k, 1],
                "Fz": F_values[k, 2],
            }
        )
    return rows, duration


def save_solution_csv(sol, stages, xs, Fs, Ts, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t_offset = 0.0
    for phase_num, (stage, x, F, T_i) in enumerate(zip(stages, xs, Fs, Ts), start=1):
        phase_rows, duration = _sample_stage_solution(sol, stage, x, F, T_i, phase_num, t_offset)
        rows.extend(phase_rows)
        t_offset += duration

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
    try:
        f = output_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        f = output_path.open("w", newline="", encoding="utf-8")

    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, output_path


def _sample_rotational_stage_solution(sol, stage, xi, torque, phase_num, t_offset, duration):
    time_grid, xi_values = sol(stage).sample(xi, grid="control")
    _, torque_values = sol(stage).sample(torque, grid="control")

    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    xi_values = _as_rows(xi_values, 7)
    torque_values = _as_rows(torque_values, 3)

    normalized_time = (time_grid - t_offset) / duration

    if torque_values.shape[0] < xi_values.shape[0]:
        pad = np.full((xi_values.shape[0] - torque_values.shape[0], 3), np.nan)
        torque_values = np.vstack([torque_values, pad])
    elif torque_values.shape[0] > xi_values.shape[0]:
        torque_values = torque_values[:xi_values.shape[0], :]

    rows = []
    for k in range(xi_values.shape[0]):
        rows.append(
            {
                "phase": phase_num,
                "node": k,
                "time": time_grid[k],
                "normalized_time": normalized_time[k],
                "T_i": duration,
                "qw": xi_values[k, 0],
                "qx": xi_values[k, 1],
                "qy": xi_values[k, 2],
                "qz": xi_values[k, 3],
                "wx": xi_values[k, 4],
                "wy": xi_values[k, 5],
                "wz": xi_values[k, 6],
                "taux": torque_values[k, 0],
                "tauy": torque_values[k, 1],
                "tauz": torque_values[k, 2],
            }
        )
    return rows


def save_rotational_solution_csv(
    sol,
    stages,
    xis,
    torques,
    phase_durations,
    output_path,
    phase_start_times=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t_offset = 0.0
    for phase_num, (stage, xi, torque, duration) in enumerate(
        zip(stages, xis, torques, phase_durations),
        start=1,
    ):
        duration = float(duration)
        phase_start = float(phase_start_times[phase_num - 1]) if phase_start_times is not None else t_offset
        rows.extend(_sample_rotational_stage_solution(sol, stage, xi, torque, phase_num, phase_start, duration))
        t_offset += duration

    fieldnames = [
        "phase",
        "node",
        "time",
        "normalized_time",
        "T_i",
        "qw",
        "qx",
        "qy",
        "qz",
        "wx",
        "wy",
        "wz",
        "taux",
        "tauy",
        "tauz",
    ]
    try:
        f = output_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        f = output_path.open("w", newline="", encoding="utf-8")

    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, output_path
