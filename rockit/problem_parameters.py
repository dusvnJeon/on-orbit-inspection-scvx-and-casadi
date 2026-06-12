from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np


def deg2rad(angle_deg: float) -> float:
    return angle_deg * pi / 180.0


@dataclass(frozen=True)
class ProblemParameters:
    """Numerical parameters from Table I of MultiphaseDDP.pdf."""

    # Deputy and orbit model
    mass: float
    inertia_diag: np.ndarray
    chief_mean_motion: float

    # Ellipsoidal keep-out / keep-in zones
    c_t_c: np.ndarray
    e_o: np.ndarray
    e_i: np.ndarray

    # LOS cone
    c_n_los: np.ndarray
    c_t_los: np.ndarray
    theta_los: float

    # Docking approach cone
    c_n_dock: np.ndarray
    c_t_dock: np.ndarray
    theta_dock: float

    # Attitude pointing / sun-avoidance parameters
    alpha: float
    d_n_b: np.ndarray
    c_w3_sun: np.ndarray

    # Bounds
    v_max: float
    omega_max: float
    f_max: float
    tau_max: float

    # Boundary conditions
    p_init: np.ndarray
    v_init: np.ndarray
    q_init: np.ndarray
    omega_init: np.ndarray
    p_final: np.ndarray
    v_final: np.ndarray
    q_final: np.ndarray
    omega_final: np.ndarray

    # Discretization / phase-time bounds
    n_phase: int
    n_grid_per_phase: int
    t_min: float
    t_max: float
    attitude_step: float
    nominal_phase_times: np.ndarray
    optimized_phase_times: np.ndarray

    @property
    def x_init(self) -> np.ndarray:
        return np.r_[self.p_init, self.v_init]

    @property
    def x_final(self) -> np.ndarray:
        return np.r_[self.p_final, self.v_final]

    @property
    def xi_init(self) -> np.ndarray:
        return np.r_[self.q_init, self.omega_init]

    @property
    def xi_final(self) -> np.ndarray:
        return np.r_[self.q_final, self.omega_final]


# ISS-like chief orbit for HCW dynamics.
# Table I does not list mean motion, so this uses a standard 400 km circular
# orbit value. Replace CHIEF_MEAN_MOTION if the paper/reviewer requires another
# orbital altitude.
EARTH_MU = 3.986004418e14  # m^3/s^2
EARTH_RADIUS = 6_378_137.0  # m
ISS_ALTITUDE = 400_000.0  # m
CHIEF_MEAN_MOTION = sqrt(EARTH_MU / (6971100.0) ** 3)   # 홍연구원님이 주신 a값 대입

PARAMS = ProblemParameters(
    mass=10.0,
    inertia_diag=np.array([0.2, 0.8, 0.2], dtype=float),
    chief_mean_motion=CHIEF_MEAN_MOTION,
    c_t_c=np.array([0.0, 0.0, 0.0], dtype=float),
    e_o=np.array([90.0, 30.0, 60.0], dtype=float),
    e_i=np.array([110.0, 110.0, 110.0], dtype=float),
    c_n_los=np.array([0.0, 0.0, 1.0], dtype=float),
    c_t_los=np.array([0.0, 0.0, 10.0], dtype=float),
    theta_los=deg2rad(20.0),
    c_n_dock=np.array([0.0, 1.0, 0.0], dtype=float),
    c_t_dock=np.array([0.0, 18.0, -0.7], dtype=float),
    theta_dock=deg2rad(45.0),
    alpha=deg2rad(60.0),
    d_n_b=np.array([0.0, 0.0, 1.0], dtype=float),
    c_w3_sun=np.array([1.0, 0.0, 0.0], dtype=float),
    v_max=2.0,
    omega_max=0.3,
    f_max=0.6,
    tau_max=0.016,
    p_init=np.array([10.0, -35.0, 5.0], dtype=float),
    v_init=np.array([0.0, 0.0, 0.0], dtype=float),
    q_init=np.array([sqrt(0.5), 0.0, 0.0, sqrt(0.5)], dtype=float),
    omega_init=np.array([0.0, 0.0, 0.0], dtype=float),
    p_final=np.array([0.0, 18.0, -0.7], dtype=float),
    v_final=np.array([0.0, 0.0, 0.0], dtype=float),
    q_final=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    omega_final=np.array([0.0, 0.0, 0.0], dtype=float),
    n_phase=4,
    n_grid_per_phase=60,
    t_min=20.0,
    t_max=300.0,
    attitude_step=0.5,
    nominal_phase_times=np.array([60.0, 60.0, 60.0, 60.0], dtype=float),
    optimized_phase_times=np.array([76.72, 55.38, 103.56, 51.74], dtype=float),
)