# on-orbit-inspection-scvx-and-casadi

봐야할 코드: 

rockit_implement_1.py : translation 문제 코드
rockit_implement_2.py : rotation 문제 코드
dynamics_residuals.py : 결과 csv로부터 residual과 cost를 요약해서 출력해주는 코드
plot_solution_histories.py : (F,v), (Tau,omega) 를 위아래로 plot해주는 코드




docs: rockit 폴더의 주요 코드와 실행 흐름 설명

rockit 폴더에 포함된 주요 최적화/분석 스크립트의 역할을 정리한다.

주요 파일 구성은 다음과 같다.

- rockit_implement_1.py: Rockit/CasADi/IPOPT 기반 translational main code
- rockit_implement_2.py: rotational main code
- cvxpy_scvx/: translational 문제를 CVXPY 기반 SCvx/SCP로 푸는 baseline 코드
- dynamics_residuals.py: dynamics residual 및 cost 계산 유틸리티
- plot_translational_solution.py: 3D translational trajectory plot 생성 코드
- plot_solution_histories.py: velocity/control 및 omega/torque history plot 생성 코드

CasADi/Rockit 결과 CSV를 warm start로 사용하여 fixed-time SCvx를 실행하는 예시는 다음과 같다.

python rockit/cvxpy_scvx/scvx_translation_baseline.py \
    --warm_start_csv rockit_outputs/translational_solution.csv \
    --solver CLARABEL

Free-phase-duration SCvx를 실행하는 예시는 다음과 같다.

python rockit/cvxpy_scvx/scvx_translation_free_time.py \
    --warm_start_csv rockit_outputs/translational_solution.csv \
    --solver CLARABEL \
    --free_time \
    --delta_T 20.0 \
    --max_iters 20

fixed-time SCvx는 warm-start CSV에서 읽은 phase duration을 고정하여 사용한다.
free-time SCvx는 phase duration을 decision variable로 두고, 현재 nominal duration 주변에서
HCW discrete dynamics를 시간에 대해 선형화하여 successive convexification 방식으로 푼다.
