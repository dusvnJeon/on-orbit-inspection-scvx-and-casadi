from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Examples:

Fixed phase durations:
    python scvx_translation_baseline.py --warm_start_csv rockit_outputs/translational_solution.csv --solver CLARABEL

Free phase durations with local SCvx time linearization:
    python scvx_translation_baseline.py \
        --warm_start_csv rockit_outputs/translational_solution.csv \
        --solver CLARABEL \
        --free_time \
        --delta_T 20.0 \
        --max_iters 20
"""


@dataclass
class ScvxParams:
    """Translational SCvx/SCP baseline parameters.

    The defaults mirror the current Rockit translational baseline in
    ``rockit/rockit_implement_1.py``. Phase times are fixed in this first
    version, so every SCvx iteration solves one convex SOCP.
    """

    m: float = 10.0
    n: float = 0.00108472
    n_phase: int = 4
    N: int = 60
    v_max: float = 2.0
    F_max: float = 0.6
    x_init: np.ndarray = field(
        default_factory=lambda: np.array([10.0, -35.0, 5.0, 0.0, 0.0, 0.0], dtype=float)
    )
    x_final: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 18.0, -0.7, 0.0, 0.0, 0.0], dtype=float)
    )
    t_c: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    e_o: np.ndarray = field(default_factory=lambda: np.array([90.0, 30.0, 60.0], dtype=float))
    e_i: np.ndarray = field(default_factory=lambda: np.array([110.0, 110.0, 110.0], dtype=float))
    n_los: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    theta_los_deg: float = 20.0
    t_los: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 10.0], dtype=float))
    n_dock: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float))
    theta_dock_deg: float = 45.0
    t_dock: np.ndarray = field(default_factory=lambda: np.array([0.0, 18.0, -0.7], dtype=float))
    phase_durations: np.ndarray = field(default_factory=lambda: np.array([60.0, 60.0, 100.0, 60.0], dtype=float))
    T_min: np.ndarray = field(default_factory=lambda: np.array([30.0, 30.0, 30.0, 30.0], dtype=float))
    T_max: np.ndarray = field(default_factory=lambda: np.array([300.0, 300.0, 300.0, 300.0], dtype=float))
    T_total_max: float = 1200.0
    delta_x: float = 50.0
    delta_u: float = 0.8
    delta_T: float = 30.0
    free_time: bool = False
    virtual_control_weight: float = 1e5
    time_weight: float = 0.0
    eps_T_fd: float = 1e-4
    initial_phase_durations: np.ndarray | None = None
    max_iters: int = 20
    change_tol: float = 1e-5
    violation_tol: float = 1e-6

    @property
    def theta_los(self) -> float:
        return math.radians(self.theta_los_deg)

    @property
    def theta_dock(self) -> float:
        return math.radians(self.theta_dock_deg)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("Cone axis must be nonzero.")
    return v / norm


def compute_hcw_Ad_Bd(h: float, n: float, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact HCW/CW zero-order-hold discretization."""

    s = math.sin(n * h)
    c = math.cos(n * h)

    Ad = np.array(
        [
            [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0],
            [6.0 * (s - n * h), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * n * h) / n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s / n],
            [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0],
            [-6.0 * n * (1.0 - c), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, -n * s, 0.0, 0.0, c],
        ],
        dtype=float,
    )

    Bd = (1.0 / m) * np.array(
        [
            [(1.0 - c) / n**2, 2.0 * (n * h - s) / n**2, 0.0],
            [-2.0 * (n * h - s) / n**2, 4.0 * (1.0 - c) / n**2 - 1.5 * h**2, 0.0],
            [0.0, 0.0, (1.0 - c) / n**2],
            [s / n, 2.0 * (1.0 - c) / n, 0.0],
            [-2.0 * (1.0 - c) / n, 4.0 * s / n - 3.0 * h, 0.0],
            [0.0, 0.0, s / n],
        ],
        dtype=float,
    )
    return Ad, Bd


def hcw_step_numpy(x: np.ndarray, u: np.ndarray, T_i: float, N: int, params: ScvxParams) -> np.ndarray:
    h = float(T_i) / N
    Ad, Bd = compute_hcw_Ad_Bd(h, params.n, params.m)
    return Ad @ x + Bd @ u


def dynamics_time_derivative_fd(
    xbar_k: np.ndarray,
    ubar_k: np.ndarray,
    Tbar_i: float,
    N: int,
    params: ScvxParams,
    eps_T: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference derivative df/dT for fixed nominal x,u.

    The perturbation is clamped so that Tbar_i - eps remains positive. This
    derivative is used only in free-time SCvx, where the nonlinear dependence
    Ad(T/N), Bd(T/N) is locally linearized around the current duration.
    """
    Tbar_i = float(Tbar_i)
    eps = min(float(eps_T), 0.49 * Tbar_i)
    if eps <= 0.0:
        raise ValueError("Tbar_i must be positive for finite-difference time linearization.")

    f_plus = hcw_step_numpy(xbar_k, ubar_k, Tbar_i + eps, N, params)
    f_minus = hcw_step_numpy(xbar_k, ubar_k, Tbar_i - eps, N, params)
    return (f_plus - f_minus) / (2.0 * eps)


def load_casadi_solution_csv(csv_path: str | Path, params: ScvxParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the Rockit/CasADi CSV format into X[phase,node,state], U[phase,k,force]."""

    df = pd.read_csv(csv_path)
    required = ["phase", "node", "T_i", "px", "py", "pz", "vx", "vy", "vz", "Fx", "Fy", "Fz"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Warm-start CSV is missing columns: {missing}")

    X = np.zeros((params.n_phase, params.N + 1, 6), dtype=float)
    U = np.zeros((params.n_phase, params.N, 3), dtype=float)
    T = np.zeros(params.n_phase, dtype=float)

    for phase in range(1, params.n_phase + 1):
        phase_df = df[df["phase"] == phase].sort_values("node")
        if len(phase_df) < params.N + 1:
            raise ValueError(f"Phase {phase} has {len(phase_df)} rows; expected at least {params.N + 1}.")

        phase_df = phase_df.iloc[: params.N + 1]
        X[phase - 1] = phase_df[["px", "py", "pz", "vx", "vy", "vz"]].to_numpy(dtype=float)
        U_phase = phase_df[["Fx", "Fy", "Fz"]].iloc[: params.N].to_numpy(dtype=float)
        U[phase - 1] = np.nan_to_num(U_phase, nan=0.0)
        T[phase - 1] = float(phase_df["T_i"].iloc[0])

    return X, U, T


def make_linear_nominal(params: ScvxParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple phase-wise linear state guess with zero control."""

    total_nodes = params.n_phase * params.N + 1
    alpha = np.linspace(0.0, 1.0, total_nodes)
    flat_x = (1.0 - alpha[:, None]) * params.x_init[None, :] + alpha[:, None] * params.x_final[None, :]

    X = np.zeros((params.n_phase, params.N + 1, 6), dtype=float)
    for phase in range(params.n_phase):
        start = phase * params.N
        X[phase] = flat_x[start : start + params.N + 1]
    U = np.zeros((params.n_phase, params.N, 3), dtype=float)
    return X, U, params.phase_durations.copy()


def _koz_linear_terms(pbar: np.ndarray, params: ScvxParams) -> tuple[float, np.ndarray]:
    scaled = (pbar - params.t_c) / params.e_o
    gbar = float(np.dot(scaled, scaled) - 1.0)
    grad = 2.0 * (pbar - params.t_c) / (params.e_o**2)
    return gbar, grad


def build_and_solve_scvx_subproblem(
    params: ScvxParams,
    xbar: np.ndarray,
    ubar: np.ndarray,
    phase_durations: np.ndarray,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> dict:
    """Build and solve one convexified translational SCvx subproblem.

    Convex constraints are imposed directly: linear dynamics, linkage,
    boundary conditions, box bounds, KIZ ellipsoid, LOS cone, docking cone,
    and trust regions. The nonconvex KOZ outside-ellipsoid constraint is
    replaced by a first-order affine lower approximation around ``xbar``.

    In fixed-time mode, exact discrete dynamics are linear because T_i is fixed.
    In free-time mode, T_i is a decision variable and the dependence
    Ad(T_i/N), Bd(T_i/N) is linearized around the current nominal Tbar_i. This
    is a local SCvx approximation, not a full nonlinear time optimization.
    """

    X = [cp.Variable((params.N + 1, 6), name=f"X_{i + 1}") for i in range(params.n_phase)]
    U = [cp.Variable((params.N, 3), name=f"U_{i + 1}") for i in range(params.n_phase)]
    T_var = cp.Variable(params.n_phase, name="T") if params.free_time else None
    Nu = [cp.Variable((params.N, 6), name=f"Nu_{i + 1}") for i in range(params.n_phase)] if params.free_time else None
    constraints = []
    objective_terms = []

    constraints.append(X[0][0, :] == params.x_init)
    constraints.append(X[-1][-1, :] == params.x_final)
    if params.free_time:
        constraints += [
            T_var >= params.T_min,
            T_var <= params.T_max,
            cp.sum(T_var) <= params.T_total_max,
            cp.abs(T_var - phase_durations) <= params.delta_T,
        ]

    los_axis = _unit(params.n_los)
    dock_axis = _unit(params.n_dock)
    los_cos = math.cos(params.theta_los)
    dock_cos = math.cos(params.theta_dock)

    for i in range(params.n_phase):
        Tbar_i = float(phase_durations[i])
        h = Tbar_i / params.N
        Ad, Bd = compute_hcw_Ad_Bd(h, params.n, params.m)

        if i < params.n_phase - 1:
            constraints.append(X[i][-1, :] == X[i + 1][0, :])

        constraints += [
            X[i][:, 3:6] <= params.v_max,
            X[i][:, 3:6] >= -params.v_max,
            U[i] <= params.F_max,
            U[i] >= -params.F_max,
            cp.norm_inf(X[i] - xbar[i]) <= params.delta_x,
            cp.norm_inf(U[i] - ubar[i]) <= params.delta_u,
        ]

        for k in range(params.N):
            if params.free_time:
                fbar = Ad @ xbar[i, k, :] + Bd @ ubar[i, k, :]
                f_T = dynamics_time_derivative_fd(
                    xbar[i, k, :],
                    ubar[i, k, :],
                    Tbar_i,
                    params.N,
                    params,
                    eps_T=params.eps_T_fd,
                )
                constraints.append(
                    X[i][k + 1, :]
                    == fbar
                    + Ad @ (X[i][k, :] - xbar[i, k, :])
                    + Bd @ (U[i][k, :] - ubar[i, k, :])
                    + f_T * (T_var[i] - Tbar_i)
                    + Nu[i][k, :]
                )
                objective_terms.append(params.virtual_control_weight * cp.norm1(Nu[i][k, :]))
            else:
                constraints.append(X[i][k + 1, :] == Ad @ X[i][k, :] + Bd @ U[i][k, :])

            # Fixed-time mode uses the exact h_i. Free-time mode uses hbar_i
            # in the convex subproblem to avoid the nonconvex product
            # T_i * ||u||^2; the actual cost is recomputed after each update.
            objective_terms.append(h * cp.sum_squares(U[i][k, :]))

        for k in range(params.N + 1):
            pos = X[i][k, 0:3]

            # Convex keep-in ellipsoid.
            constraints.append(cp.norm(cp.multiply(1.0 / params.e_i, pos - params.t_c), 2) <= 1.0)

            # Nonconvex keep-out ellipsoid, linearized as an affine constraint.
            if i != 3:
                gbar, grad = _koz_linear_terms(xbar[i, k, 0:3], params)
                constraints.append(gbar + grad @ (pos - xbar[i, k, 0:3]) >= 0.0)

            # Convex second-order cone for LOS visibility in phase 2.
            if i == 1:
                r = pos - params.t_los
                constraints.append(los_axis @ r >= 0.0)
                constraints.append(los_cos * cp.norm(r, 2) <= los_axis @ r)

            # Convex second-order cone for docking approach in phase 4.
            if i == 3:
                r = pos - params.t_dock
                constraints.append(dock_axis @ r >= 0.0)
                constraints.append(dock_cos * cp.norm(r, 2) <= dock_axis @ r)

    if params.free_time and params.time_weight != 0.0:
        objective_terms.append(params.time_weight * cp.sum(T_var))

    problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)

    solve_kwargs = {"solver": solver.upper(), "verbose": verbose, "warm_start": True}
    if solver.upper() == "CLARABEL":
        solve_kwargs["max_iter"] = 1000

    tic = time.perf_counter()
    try:
        problem.solve(**solve_kwargs)
    except cp.error.SolverError:
        if solver.upper() != "CLARABEL" and "CLARABEL" in cp.installed_solvers():
            solve_kwargs["solver"] = "CLARABEL"
            problem.solve(**solve_kwargs)
        else:
            raise
    solve_time = time.perf_counter() - tic

    x_val = np.array([Xi.value for Xi in X], dtype=float) if X[0].value is not None else None
    u_val = np.array([Ui.value for Ui in U], dtype=float) if U[0].value is not None else None
    t_val = np.asarray(T_var.value, dtype=float) if params.free_time and T_var.value is not None else phase_durations.copy()
    nu_val = np.array([Nui.value for Nui in Nu], dtype=float) if params.free_time and Nu[0].value is not None else None
    return {
        "status": problem.status,
        "objective": problem.value,
        "solve_time": solve_time,
        "X": x_val,
        "U": u_val,
        "T": t_val,
        "Nu": nu_val,
    }


def compute_constraint_metrics(
    params: ScvxParams,
    X: np.ndarray,
    U: np.ndarray,
    phase_durations: np.ndarray,
) -> dict[str, float]:
    """Return positive violation magnitudes for the original constraints."""

    los_axis = _unit(params.n_los)
    dock_axis = _unit(params.n_dock)
    los_cos = math.cos(params.theta_los)
    dock_cos = math.cos(params.theta_dock)

    max_kiz = 0.0
    max_koz = 0.0
    max_los = 0.0
    max_dock = 0.0
    max_dyn = 0.0

    for i in range(params.n_phase):
        h = float(phase_durations[i]) / params.N
        Ad, Bd = compute_hcw_Ad_Bd(h, params.n, params.m)
        for k in range(params.N):
            max_dyn = max(max_dyn, float(np.linalg.norm(X[i, k + 1] - (Ad @ X[i, k] + Bd @ U[i, k]), ord=np.inf)))

        for k in range(params.N + 1):
            pos = X[i, k, 0:3]
            kiz = np.linalg.norm((pos - params.t_c) / params.e_i) - 1.0
            max_kiz = max(max_kiz, float(kiz))

            if i != 3:
                koz_g = float(np.sum(((pos - params.t_c) / params.e_o) ** 2) - 1.0)
                max_koz = max(max_koz, -koz_g)

            if i == 1:
                r = pos - params.t_los
                los_v = los_cos * np.linalg.norm(r) - float(los_axis @ r)
                max_los = max(max_los, los_v, -float(los_axis @ r))

            if i == 3:
                r = pos - params.t_dock
                dock_v = dock_cos * np.linalg.norm(r) - float(dock_axis @ r)
                max_dock = max(max_dock, dock_v, -float(dock_axis @ r))

    linkage = 0.0
    for i in range(params.n_phase - 1):
        linkage = max(linkage, float(np.linalg.norm(X[i, -1] - X[i + 1, 0], ord=np.inf)))

    return {
        "max_kiz_violation": max(0.0, max_kiz),
        "max_koz_violation": max(0.0, max_koz),
        "max_los_violation": max(0.0, max_los),
        "max_docking_violation": max(0.0, max_dock),
        "max_velocity_violation": max(0.0, float(np.max(np.abs(X[:, :, 3:6])) - params.v_max)),
        "max_force_violation": max(0.0, float(np.max(np.abs(U)) - params.F_max)),
        "terminal_error": float(np.linalg.norm(X[-1, -1] - params.x_final, ord=np.inf)),
        "linkage_residual": linkage,
        "dynamics_residual": max_dyn,
    }


def compute_actual_costs(params: ScvxParams, X: np.ndarray, U: np.ndarray, T: np.ndarray) -> dict[str, float]:
    """Compute costs and simple extrema using the current exact durations."""
    squared_force_cost = 0.0
    fuel_like_cost = 0.0
    for i in range(params.n_phase):
        h = float(T[i]) / params.N
        for k in range(params.N):
            u_norm = float(np.linalg.norm(U[i, k, :], ord=2))
            squared_force_cost += h * u_norm**2
            fuel_like_cost += h * u_norm

    return {
        "squared_force_cost": squared_force_cost,
        "fuel_like_cost": fuel_like_cost,
        "max_force": float(np.max(np.abs(U))),
        "max_velocity": float(np.max(np.abs(X[:, :, 3:6]))),
    }


def _max_original_violation(metrics: dict[str, float]) -> float:
    keys = [
        "max_kiz_violation",
        "max_koz_violation",
        "max_los_violation",
        "max_docking_violation",
        "max_velocity_violation",
        "max_force_violation",
        "terminal_error",
        "linkage_residual",
        "dynamics_residual",
    ]
    return max(metrics[k] for k in keys)


def run_scvx(
    params: ScvxParams,
    warm_start_csv: str | Path | None = None,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> dict:
    if warm_start_csv:
        xbar, ubar, phase_durations = load_casadi_solution_csv(warm_start_csv, params)
    else:
        xbar, ubar, phase_durations = make_linear_nominal(params)
    if params.initial_phase_durations is not None:
        phase_durations = params.initial_phase_durations.copy()

    history = []
    latest = None
    for iteration in range(1, params.max_iters + 1):
        latest = build_and_solve_scvx_subproblem(params, xbar, ubar, phase_durations, solver=solver, verbose=verbose)
        if latest["X"] is None or latest["U"] is None or latest["status"] not in {"optimal", "optimal_inaccurate"}:
            print(f"{iteration:02d} | status={latest['status']} | no primal solution returned")
            if params.free_time:
                print(
                    "Free-time diagnostic: try changing --delta_T. Increasing it can relax duration movement; "
                    "decreasing it can improve a poor time linearization. Virtual-control weight changes the "
                    "objective penalty, not feasibility."
                )
            break

        X_new = latest["X"]
        U_new = latest["U"]
        T_new = latest["T"]
        metrics = compute_constraint_metrics(params, X_new, U_new, T_new)
        costs = compute_actual_costs(params, X_new, U_new, T_new)
        max_change = max(float(np.max(np.abs(X_new - xbar))), float(np.max(np.abs(U_new - ubar))))
        max_T_change = float(np.max(np.abs(T_new - phase_durations)))
        max_violation = _max_original_violation(metrics)

        row = {
            "iteration": iteration,
            "status": latest["status"],
            "objective": float(latest["objective"]),
            "actual_J": costs["squared_force_cost"],
            "fuel_J": costs["fuel_like_cost"],
            "solve_time": float(latest["solve_time"]),
            "max_change": max_change,
            "max_T_change": max_T_change,
            "max_violation": max_violation,
            "T": T_new.copy(),
            **metrics,
            **costs,
        }
        history.append(row)
        print_summary_row(row)

        xbar, ubar, phase_durations = X_new, U_new, T_new
        if max_change <= params.change_tol and max_T_change <= params.change_tol and max_violation <= params.violation_tol:
            break

    return {
        "X": xbar,
        "U": ubar,
        "phase_durations": phase_durations,
        "history": history,
        "last_status": latest["status"] if latest else None,
    }


def save_solution_csv(
    X: np.ndarray,
    U: np.ndarray,
    phase_durations: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t_offset = 0.0
    for i, duration in enumerate(phase_durations):
        duration = float(duration)
        for k in range(X.shape[1]):
            tau = k / (X.shape[1] - 1)
            force = U[i, k] if k < U.shape[1] else np.full(3, np.nan)
            rows.append(
                {
                    "phase": i + 1,
                    "node": k,
                    "time": t_offset + tau * duration,
                    "tau": tau,
                    "T_i": duration,
                    "px": X[i, k, 0],
                    "py": X[i, k, 1],
                    "pz": X[i, k, 2],
                    "vx": X[i, k, 3],
                    "vy": X[i, k, 4],
                    "vz": X[i, k, 5],
                    "Fx": force[0],
                    "Fy": force[1],
                    "Fz": force[2],
                }
            )
        t_offset += duration

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def plot_translation_result(X: np.ndarray, output_path: str | Path | None = None) -> None:
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i in range(X.shape[0]):
        ax.plot(X[i, :, 0], X[i, :, 1], X[i, :, 2], color=colors[i % len(colors)], label=f"Phase {i + 1}")
        ax.scatter(X[i, 0, 0], X[i, 0, 1], X[i, 0, 2], color=colors[i % len(colors)], s=18)
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_zlabel("pz [m]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary_header() -> None:
    print(
        "it | status             | objective    | actual_J     | fuel_J       | solve[s] | "
        "dxdu     | dT       | max viol | T1       T2       T3       T4"
    )


def print_summary_row(row: dict) -> None:
    T = row["T"]
    print(
        f"{row['iteration']:02d} | {row['status']:<18} | {row['objective']:10.3e} | "
        f"{row['actual_J']:10.3e} | {row['fuel_J']:10.3e} | {row['solve_time']:8.3f} | "
        f"{row['max_change']:8.2e} | {row['max_T_change']:8.2e} | {row['max_violation']:8.2e} | "
        f"{T[0]:8.3f} {T[1]:8.3f} {T[2]:8.3f} {T[3]:8.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCvx/SCP translational CVXPY baseline with optional free phase time.")
    parser.add_argument("--warm_start_csv", type=str, default=None)
    parser.add_argument("--solver", type=str, default="CLARABEL", help="CLARABEL or MOSEK if installed.")
    parser.add_argument("--output_csv", type=str, default="rockit_outputs/scvx_free_time_translation_solution.csv")
    parser.add_argument("--plot_path", type=str, default="rockit_outputs/scvx_free_time_translation_trajectory.png")
    parser.add_argument("--max_iters", type=int, default=20)
    parser.add_argument("--delta_x", type=float, default=50.0)
    parser.add_argument("--delta_u", type=float, default=0.8)
    parser.add_argument("--free_time", action="store_true")
    parser.add_argument("--delta_T", type=float, default=30.0)
    parser.add_argument("--T_total_max", type=float, default=1200.0)
    parser.add_argument("--virtual_control_weight", type=float, default=1e5)
    parser.add_argument("--time_weight", type=float, default=0.0)
    parser.add_argument("--eps_T_fd", type=float, default=1e-4)
    parser.add_argument(
        "--initial_phase_durations",
        type=float,
        nargs=4,
        default=None,
        metavar=("T1", "T2", "T3", "T4"),
        help="Override the initial nominal phase durations, even when warm-starting X/U from CSV.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ScvxParams(
        max_iters=args.max_iters,
        delta_x=args.delta_x,
        delta_u=args.delta_u,
        free_time=args.free_time,
        delta_T=args.delta_T,
        T_total_max=args.T_total_max,
        virtual_control_weight=args.virtual_control_weight,
        time_weight=args.time_weight,
        eps_T_fd=args.eps_T_fd,
        initial_phase_durations=(
            np.array(args.initial_phase_durations, dtype=float) if args.initial_phase_durations is not None else None
        ),
    )

    print_summary_header()
    result = run_scvx(params, warm_start_csv=args.warm_start_csv, solver=args.solver, verbose=args.verbose)
    csv_path = save_solution_csv(result["X"], result["U"], result["phase_durations"], args.output_csv)
    plot_translation_result(result["X"], args.plot_path)

    final_metrics = compute_constraint_metrics(params, result["X"], result["U"], result["phase_durations"])
    final_costs = compute_actual_costs(params, result["X"], result["U"], result["phase_durations"])
    print("\nFinal metrics")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.6e}")
    print("\nFinal costs")
    for key, value in final_costs.items():
        print(f"  {key}: {value:.6e}")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {args.plot_path}")


if __name__ == "__main__":
    main()
