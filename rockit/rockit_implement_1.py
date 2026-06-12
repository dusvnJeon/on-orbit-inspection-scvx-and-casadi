from rockit import Ocp, MultipleShooting
import casadi as ca
from dynamics_rollout import *
from pathlib import Path

from problem_parameters import PARAMS
from solution_io import save_solution_csv

"""
현재 문제
1. mean motion 수치값 부탁드립니다.
2. los cone과 docking cone의 반각이 논문 표에서 구분되지 않음 (둘다 theta_los로 적혀있음)
3. 논문 표 상에서 p_final과 p_initial이 동일함. 논문의 figure를 보면 둘은 다른 위치에 있는 것같음.
4. Switching time, x, u에 대한 initial guess가 있을까요?
"""

"""
Debugging note:
1. Los cone과 docking cone에 대해 r의 norm이 들어가면 이에 대한 미분에서 r의 norm이 0이 되면서 문제가 생김.
    -> r의 norm 대신 r의 제곱이 들어가도록 수정. 그러면 double cover라서 부호 제약을 통해 한쪽만 커버하게 함.
"""

"""
혹시...
T_max가 걸리는게 모든 시간에 대한 합인가요...? 각각의 phase time이 아니라...?
"""


p = PARAMS
m = p.mass
n = p.chief_mean_motion

N = p.n_grid_per_phase
T_max = p.t_max
T_min = p.t_min

# KOZ / KIZ parameters
E_o = p.e_o                 # KOZ의 타원 파라미터 (나눠줘야함)
E_i = p.e_i                 # KIZ의 타원 파라미터 (나눠줘야함)

c_t_c = p.c_t_c             # cheif 기저에서 본 cheif의 위치 벡터

# los / docking cone parameters
c_n_los = p.c_n_los         # cheif 기저에서 본 los cone의 중심축 단위 벡터
c_t_los = p.c_t_los         # cheif 기저에서 본 los cone의 꼭짓점 위치 벡터
theta_los = p.theta_los     # los cone 반각

c_n_dock = p.c_n_dock       # cheif 기저에서 본 docking cone의 중심축 단위 벡터
c_t_dock = p.c_t_dock       # cheif 기저에서 본 docking cone의 꼭짓점 위치 벡터
theta_dock = p.theta_dock   # docking cone 반각

# bounds
v_max = p.v_max
F_max = p.f_max

x_init = p.x_init
x_final = p.x_final

# 일단 translational dynamics에 필요한 변수만 선언

def create_translational_stage(ocp, phase_num):
    """
    Create translational stage for each Phases.
    
    considering pahse_num(= 1,2,3,4) that gives evidence of which constraints have to be added
    Phase 1(Moving Phase) :  initial state constraints
    Phase 2(Los Phase) : LOS cone constraints
    Phase 3(Moving Phase)
    Phase 4(Docking Phase) : Docking cone constraints, final state constraints, except KOZ
    
    And all of Phases includes 
    (i) box constraints for v and F
    (ii) dynamics constraints
    (iii) KOZ (but except for Phase 4)
    (iv) KIZ
    (v) T_min < T_i < T_max

    And each stage connect with each other by the continuity constraints. 
        This will be added outside of this function.
        Initial state and final state constraints also be.
    """

    # 여기서의 T는 더미 인덱스고, 실제로 Switching time은 T_i로 표현됨. state에 T_i를 넣어서 set_next에서 자유도가 남는 문제 해결을 위함
    stage = ocp.stage(t0=0, T=1)        # terminal time of each stage는 initial guess를 뭐로 설정하셨을까?


    x = stage.state(6)
    T_i = stage.state()
    F = stage.control(3)

    x_next = hcw_step_casadi(x,F,T_i,N,n,m)
        # x_next = hcw_step_casadi(x,F,T,N,n,m)

    # Dynamics constraints
    stage.set_next(x, x_next)
    stage.set_next(T_i, T_i)

    pos = x[0:3]
    vel = x[3:6]

    # Box constraints for v and F
        # 예제는 v_min, F_min 제약조건은 없음
    stage.subject_to(-v_max <= vel)
    stage.subject_to(vel <= v_max)
    stage.subject_to(-F_max <= F)
    stage.subject_to(F <= F_max)

    # KIZ
    d = pos - ca.DM(c_t_c)
    kiz_value = ca.sumsqr(d / ca.DM(p.e_i))
    stage.subject_to(kiz_value <= 1)

    # KOZ
    if phase_num != 4:
        koz_value = ca.sumsqr(d / ca.DM(p.e_o))
        stage.subject_to(koz_value >= 1)

    # LOS cone
    if phase_num == 2:
        r = pos - ca.DM(c_t_los)
        axis = ca.DM(c_n_los)
        # stage.subject_to(ca.dot(r, axis) >= ca.norm_2(r) * ca.cos(theta_los))
        proj = ca.dot(r, axis)
        stage.subject_to(proj >= 0)
        stage.subject_to(proj**2 >= (ca.cos(theta_los)**2) * ca.sumsqr(r))

    # Docking cone
    if phase_num == 4:
        r = pos - ca.DM(c_t_dock)
        axis = ca.DM(c_n_dock)
        # stage.subject_to(ca.dot(r, axis) >= ca.norm_2(r) * ca.cos(theta_dock))
        proj = ca.dot(r, axis)
        stage.subject_to(proj >= 0)
        stage.subject_to(proj**2 >= (ca.cos(theta_dock)**2) * ca.sumsqr(r))

    # Time box constraints
    stage.subject_to(T_i >= 30.0)
    stage.subject_to(T_i <= 300)


    # stage.subject_to(stage.T >= T_min)
    # stage.subject_to(stage.T <= T_max)
    stage.set_initial(T_i, 300)

    stage.method(MultipleShooting(N=N))
    return stage, x, F, T_i

ocp = Ocp()

# Phase1 : Moving Phase
stage1, x1, F1, T1 = create_translational_stage(ocp, phase_num=1)
ocp.subject_to(stage1.t0 == 0)  # Stage starts at time 0 
ocp.subject_to(stage1.at_t0(x1) == ca.DM(x_init))

# Phase2 : Inspection Phase
stage2, x2, F2, T2 = create_translational_stage(ocp, phase_num=2)
ocp.subject_to(stage1.at_tf(x1) == stage2.at_t0(x2))

# Phase3 : Moving Phase
stage3, x3, F3, T3 = create_translational_stage(ocp, phase_num=3)
ocp.subject_to(stage2.at_tf(x2) == stage3.at_t0(x3))

# Phase4 : Docking Phase
stage4, x4, F4, T4 = create_translational_stage(ocp, phase_num=4)
ocp.subject_to(stage3.at_tf(x3) == stage4.at_t0(x4))
ocp.subject_to(stage4.at_tf(x4) == ca.DM(x_final))

# Time Max
# ocp.subject_to(stage1.at_t0(T1) + stage2.at_t0(T2) + stage3.at_t0(T3) + stage4.at_t0(T4) <= T_max)


# objective (minimize fuel)
eps = 1e-6
# stage1.add_objective(stage1.sum((T1 / N) * ca.sqrt(ca.sumsqr(F1) + eps)))
# stage2.add_objective(stage2.sum((T2 / N) * ca.sqrt(ca.sumsqr(F2) + eps)))
# stage3.add_objective(stage3.sum((T3 / N) * ca.sqrt(ca.sumsqr(F3) + eps)))
# stage4.add_objective(stage4.sum((T4 / N) * ca.sqrt(ca.sumsqr(F4) + eps)))

stage1.add_objective(stage1.sum((T1 / N) * (ca.sumsqr(F1) + eps)))
stage2.add_objective(stage2.sum((T2 / N) * (ca.sumsqr(F2) + eps)))
stage3.add_objective(stage3.sum((T3 / N) * (ca.sumsqr(F3) + eps)))
stage4.add_objective(stage4.sum((T4 / N) * (ca.sumsqr(F4) + eps)))

ocp.solver(
    "ipopt",
    {
        "ipopt.max_iter": 10000,
        "ipopt.print_level": 5,
    },
)
sol = ocp.solve()

# ======================================= CSV Saving ==========================================
stages = [stage1, stage2, stage3, stage4]
xs = [x1, x2, x3, x4]
Fs = [F1, F2, F3, F4]
Ts = [T1, T2, T3, T4]

output_dir = Path("rockit_outputs")
_, csv_path = save_solution_csv(sol, stages, xs, Fs, Ts, output_dir / "translational_solution.csv")
print(f"Saved {csv_path}")
