import casadi as ca


def hcw_Ad_Bd_casadi(h, n, m):
    """
    CasADi symbolic HCW exact ZOH discretization.

    x = [px, py, pz, vx, vy, vz]^T
    u = [Fx, Fy, Fz]^T

    x_next = Ad(h) @ x + Bd(h) @ u

    h can be a CasADi symbolic variable.
    """
    c = ca.cos(n * h)
    s = ca.sin(n * h)

    Ad = ca.vertcat(
        ca.horzcat(4 - 3*c,        0, 0, s/n,        2*(1-c)/n,       0),
        ca.horzcat(6*(s - n*h),    1, 0, -2*(1-c)/n, (4*s - 3*n*h)/n, 0),
        ca.horzcat(0,              0, c, 0,          0,               s/n),
        ca.horzcat(3*n*s,          0, 0, c,          2*s,             0),
        ca.horzcat(-6*n*(1-c),     0, 0, -2*s,       4*c - 3,         0),
        ca.horzcat(0,              0, -n*s, 0,       0,               c),
    )

    Bd = (1 / m) * ca.vertcat(
        ca.horzcat((1-c)/n**2,        2*(n*h - s)/n**2,           0),
        ca.horzcat(-2*(n*h - s)/n**2, 4*(1-c)/n**2 - 1.5*h**2,    0),
        ca.horzcat(0,                 0,                          (1-c)/n**2),
        ca.horzcat(s/n,               2*(1-c)/n,                  0),
        ca.horzcat(-2*(1-c)/n,        4*s/n - 3*h,                0),
        ca.horzcat(0,                 0,                          s/n),
    )

    return Ad, Bd


def hcw_step_casadi(x, u, T_i, N_i, n, m):
    """
    One-step symbolic rollout with decision variable phase duration T_i.

    h_i = T_i / N_i
    x_next = Ad(h_i) x + Bd(h_i) u
    """
    h_i = T_i / N_i
    Ad, Bd = hcw_Ad_Bd_casadi(h_i, n, m)
    return Ad @ x + Bd @ u


def q2cRd(q):
    """
    Direction cosine matrix from deputy/body frame to chief/world frame.

    q = [qw, qx, qy, qz]^T.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    return ca.vertcat(
        ca.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
        ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)),
    )


def quaternion_derivative_casadi(xi, tau, J):
    """
    Continuous-time rigid-body attitude dynamics.

    xi = [qw, qx, qy, qz, wx, wy, wz]^T.
    tau = [tx, ty, tz]^T.
    J is the diagonal inertia vector [Jx, Jy, Jz].
    """
    q = xi[0:4]
    w = xi[4:7]
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    Lq = ca.vertcat(
        ca.horzcat(-q1, -q2, -q3),
        ca.horzcat(q0, -q3, q2),
        ca.horzcat(q3, q0, -q1),
        ca.horzcat(-q2, q1, q0),
    )
    q_dot = 0.5 * Lq @ w

    J_dm = ca.DM(J)
    Jw = J_dm * w
    w_dot = (tau - ca.cross(w, Jw)) / J_dm

    return ca.vertcat(q_dot, w_dot)


def quat_mul_casadi(q, p):
    """
    Quaternion product for scalar-first quaternions.

    q = [qw, qx, qy, qz]^T.
    p = [pw, px, py, pz]^T.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    pw, px, py, pz = p[0], p[1], p[2], p[3]

    return ca.vertcat(
        qw * pw - qx * px - qy * py - qz * pz,
        qw * px + qx * pw + qy * pz - qz * py,
        qw * py - qx * pz + qy * pw + qz * px,
        qw * pz + qx * py - qy * px + qz * pw,
    )


def quat_exp_casadi(delta_theta):
    """
    Exponential map from a rotation vector to a scalar-first unit quaternion.

    delta_theta is a 3-vector. The result is
    [cos(||delta||/2), sin(||delta||/2) * delta / ||delta||].
    """
    angle_sq = ca.sumsqr(delta_theta)
    angle = ca.sqrt(angle_sq + 1e-16)
    half = 0.5 * angle

    scale = ca.sin(half) / angle

    return ca.vertcat(ca.cos(half), scale * delta_theta)


def attitude_step_lie_casadi(xi, tau, h, J):
    """
    One-step Lie-group attitude update.

    xi = [qw, qx, qy, qz, wx, wy, wz]^T.
    The quaternion is advanced as q_next = q * exp(h*w), which preserves
    unit norm when q is unit norm.
    """
    q = xi[0:4]
    w = xi[4:7]

    Jmat = ca.diag(ca.DM(J))
    Jw = Jmat @ w
    w_dot = ca.solve(Jmat, tau - ca.cross(w, Jw))
    w_next = w + h * w_dot

    dq = quat_exp_casadi(h * w)
    q_next = quat_mul_casadi(q, dq)

    return ca.vertcat(q_next, w_next)
