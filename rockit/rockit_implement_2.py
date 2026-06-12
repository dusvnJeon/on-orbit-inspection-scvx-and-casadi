from rockit import Ocp, MultipleShooting
import casadi as ca
from dynamics_rollout import *
import numpy as np
from pathlib import Path
from math import floor

from problem_parameters import PARAMS
from solution_io import save_rotational_solution_csv

p = PARAMS
J = p.inertia_diag
ATTITUDE_STEP = p.attitude_step


c_t_c = p.c_t_c             # cheif 기저에서 본 cheif의 위치 벡터

# los / docking cone parameters
c_n_los = p.c_n_los         # cheif 기저에서 본 los cone의 중심축 단위 벡터
theta_los = p.theta_los     # los cone 반각

c_n_dock = p.c_n_dock       # cheif 기저에서 본 docking cone의 중심축 단위 벡터
theta_dock = p.theta_dock   # docking cone 반각

# camera parameters
# d_n_b = p.d_n_b        # deputy 기저에서 본 카메라 시야의 중심축 단위 벡터 (deputy의 z축과 일치)
d_n_b = np.array([0.0, 1.0, 0.0], dtype=float)
alpha = p.alpha             # 카메라 시야 반각

c_w3_sun = p.c_w3_sun       # Sun avoiding을 위한 sun 벡터

# bounds
omega_max = p.omega_max
tau_max = p.tau_max

xi_init = p.xi_init
xi_final = p.xi_final


# load csv data from solved translational OCP
def load_phase_times_from_csv(csv_path):
    import csv

    phase_times = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phase_times.setdefault(int(row["phase"]), float(row["T_i"]))

    phase_durations = [phase_times[i] for i in range(1, 5)]
    if any(T <= 0 for T in phase_durations):
        raise ValueError(f"Phase durations must be positive: {phase_durations}")
    return phase_durations

# processing data to use on rotational OCP
def attitude_phase_schedule_from_translation(phase_durations, h=ATTITUDE_STEP):
    """
    Build a fixed-step attitude schedule from translational phase durations.

    Translational phase durations are rounded down to integer seconds.
    For example, if phase 1 ends at 87.4132 s, attitude phase 1 ends at
    87.0 s and attitude phase 2 also starts at 87.0 s. The first interval
    of phase 2 then runs from 87.0 s to 87.5 s.
    """
    attitude_durations = [float(floor(T)) for T in phase_durations]
    phase_grid_counts = []
    for T in attitude_durations:
        N_i = int(round(T / h))
        if N_i <= 0 or abs(N_i * h - T) > 1e-12:
            raise ValueError(f"Attitude duration {T} is not compatible with h={h}.")
        phase_grid_counts.append(N_i)

    phase_start_times = [0.0]
    for T in attitude_durations[:-1]:
        phase_start_times.append(phase_start_times[-1] + T)

    return attitude_durations, phase_grid_counts, phase_start_times



# ======================================= rotation problem ==========================================


def create_rotational_stage(ocp, phase_num, phase_start_times, phase_durations, phase_grid_counts):
    """
    Create rotational stage for each Phases.
    
    considering pahse_num(= 1,2,3,4) that gives evidence of which constraints have to be added
    Phase 1(Moving Phase) :  initial state constraints
    Phase 2(Los Phase) : FOV inclusion (los)
    Phase 3(Moving Phase)
    Phase 4(Docking Phase) : FOV inclusion (docking), final state constraints
    
    And all of Phases includes 
    (i) box constraints for omega and tau
    (ii) Sun avoiding

    And each stage connect with each other by the continuity constraints. 
        This will be added outside of this function.
        Initial state and final state constraints also be.

    * Switching Time is already determined by transitional OCP solving.
        -> Switching Time is excluded for decision variables.
    """

    if phase_num not in (1, 2, 3, 4):
        raise ValueError(f"No such phase: {phase_num}")

    t0 = phase_start_times[phase_num - 1]
    T = phase_durations[phase_num - 1]
    N_i = phase_grid_counts[phase_num - 1]
    stage = ocp.stage(t0=t0, T=T)


    xi = stage.state(7)  # quaternion and omega
    tau = stage.control(3)

    xi_next = attitude_step_lie_casadi(xi, tau, ATTITUDE_STEP, J)

    # Discrete Lie-group dynamics constraints
    stage.set_next(xi, xi_next)

    q = xi[0:4]
    omega = xi[4:7]

    # Box constraints for omega and tau
    stage.subject_to(-omega_max <= omega)
    stage.subject_to(omega <= omega_max)
    stage.subject_to(-tau_max <= tau)
    stage.subject_to(tau <= tau_max)

    # Sun avoiding : FOV exclusive
    camera_axis = q2cRd(q) @ ca.DM(d_n_b)
    stage.subject_to(ca.dot(ca.DM(c_w3_sun), camera_axis) <= ca.cos(alpha))

    # Inspecting FOV inclusicve
    if phase_num == 2:
        stage.subject_to(-ca.dot(ca.DM(c_n_los), camera_axis) >= ca.cos(alpha - theta_los))

    # Docking FOV inclusicve
    if phase_num == 4:
        stage.subject_to(-ca.dot(ca.DM(c_n_dock), camera_axis) >= ca.cos(alpha - theta_dock))

    stage.method(MultipleShooting(N=N_i))
    return stage, xi, tau

ocp = Ocp()

csv_path = Path("rockit_outputs") / "translational_solution.csv"
translation_phase_durations = load_phase_times_from_csv(csv_path)
phase_durations, phase_grid_counts, phase_start_times = attitude_phase_schedule_from_translation(
    translation_phase_durations
)


# Phase1 : Moving Phase
stage1, xi1, tau1 = create_rotational_stage(
    ocp,
    phase_num=1,
    phase_start_times=phase_start_times,
    phase_durations=phase_durations,
    phase_grid_counts=phase_grid_counts,
)
ocp.subject_to(stage1.at_t0(xi1) == ca.DM(xi_init))

# Phase2 : Inspection Phase
stage2, xi2, tau2 = create_rotational_stage(
    ocp,
    phase_num=2,
    phase_start_times=phase_start_times,
    phase_durations=phase_durations,
    phase_grid_counts=phase_grid_counts,
)
ocp.subject_to(stage1.at_tf(xi1) == stage2.at_t0(xi2))

# Phase3 : Moving Phase
stage3, xi3, tau3 = create_rotational_stage(
    ocp,
    phase_num=3,
    phase_start_times=phase_start_times,
    phase_durations=phase_durations,
    phase_grid_counts=phase_grid_counts,
)
ocp.subject_to(stage2.at_tf(xi2) == stage3.at_t0(xi3))

# Phase4 : Docking Phase
stage4, xi4, tau4 = create_rotational_stage(
    ocp,
    phase_num=4,
    phase_start_times=phase_start_times,
    phase_durations=phase_durations,
    phase_grid_counts=phase_grid_counts,
)
ocp.subject_to(stage3.at_tf(xi3) == stage4.at_t0(xi4))
ocp.subject_to(stage4.at_tf(xi4) == ca.DM(xi_final))


# objective (minimize control effort)
eps = 1e-8
for stage, tau, T in [
    (stage1, tau1, phase_durations[0]),
    (stage2, tau2, phase_durations[1]),
    (stage3, tau3, phase_durations[2]),
    (stage4, tau4, phase_durations[3]),
]:
    # 연료 norm 최소화
    stage.add_objective(ATTITUDE_STEP * stage.sum(ca.sqrt(ca.sumsqr(tau) + eps)))

# initial guess 설정. 근데 tau는 다 0으로 두고, quaternion은 phase 1,2는 초기값, 2,3은 끝값 사용
for stage, xi, tau, xi_guess in [
    (stage1, xi1, tau1, xi_init),
    (stage2, xi2, tau2, xi_init),
    (stage3, xi3, tau3, xi_final),
    (stage4, xi4, tau4, xi_final),
]:
    stage.set_initial(xi, ca.DM(xi_guess))
    stage.set_initial(tau, ca.DM.zeros(3, 1))

ocp.solver(
    "ipopt",
    {
        "ipopt.max_iter": 10000,
        "ipopt.print_level": 5,
    },
)

        # "ipopt.acceptable_tol": 1e-5,
        # "ipopt.acceptable_iter": 20,
        # "ipopt.mumps_mem_percent": 5000,

sol = ocp.solve()

stages = [stage1, stage2, stage3, stage4]
xis = [xi1, xi2, xi3, xi4]
torques = [tau1, tau2, tau3, tau4]

output_dir = Path("rockit_outputs")
_, csv_path = save_rotational_solution_csv(
    sol,
    stages,
    xis,
    torques,
    phase_durations,
    output_dir / "rotational_solution.csv",
    phase_start_times=phase_start_times,
)
print(f"Saved {csv_path}")
