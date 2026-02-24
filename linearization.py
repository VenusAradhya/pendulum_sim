import numpy as np
from equations_of_motion import equations_of_motion, M1, M2, L1, L2, G

# Numerical Linearisation
def linearise_numerical():
    """
    LQR requires a *linear* model of the system: ẋ = A·x + B·u
    But our pendulum EOM are nonlinear
    So we linearise — we approximate them as linear near the
    downward equilibrium [th1=0, th2=0, w1=0, w2=0].

    We do this numerically: slightly nudge each state variable by a tiny
    amount (eps), run the physics, and measure how much the output changes.
    This is just a numerical derivative (finite difference).

    Returns:
        A (4×4): how the state evolves on its own without any control
        B (4×1): how the control force u influences the state
    """
    x0  = np.zeros(4)  # equilibrium point: everything at zero
    eps = 1e-6          # tiny nudge for numerical differentiation

    # A = ∂f/∂x — perturb each state variable one at a time
    A = np.zeros((4, 4))
    for i in range(4):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps; xm[i] -= eps
        # x_p_ddot=0 and force_val=0 at equilibrium — only perturbing state here
        A[:, i] = (equations_of_motion(xp, 0.0, 0.0) - equations_of_motion(xm, 0.0, 0.0)) / (2*eps)

    # B = ∂f/∂force_val — perturb the control force input only, keep x_p_ddot=0
    B = ((equations_of_motion(x0, 0.0, eps) - equations_of_motion(x0, 0.0, -eps)) / (2*eps)).reshape(4, 1)

    return A, B

# Analytical (By-Hand) Linearisation
def linearise_analytical():
    """
    Closed-form linearisation derived by hand using small-angle approximations:
        sin(θ) ≈ θ,  cos(θ) ≈ 1,  sin(δ) ≈ δ = θ1−θ2
        ω² terms → 0  (second-order small)

    Under these approximations the denominator simplifies:
        D = 2M1 + M2 − M2·cos(2δ) → 2M1 + M2 − M2·1 = 2M1

    Linearised θ̈1 (dropping all quadratic terms):
        numerator ≈ −G(2M1+M2)θ1 − M2·G·(θ1−2θ2) + force_val·L1
        → −G(2M1+2M2)θ1 + 2M2·G·θ2 + force_val·L1

    Dividing by L1·2M1:
        θ̈1 = −[G(M1+M2)/(M1·L1)]·θ1 + [M2·G/(M1·L1)]·θ2 + force_val/(2M1·L1)

    Linearised θ̈2:
        numerator ≈ 2(θ1−θ2)·G(M1+M2)

    Dividing by L2·2M1:
        θ̈2 = [G(M1+M2)/(M1·L2)]·θ1 − [G(M1+M2)/(M1·L2)]·θ2

    Note: x_p_ddot does not appear in A or B because it enters as an
    external disturbance, not as a state or control input. The LQR is
    designed to reject it via state feedback alone.

    This function should give the same result as linearise_numerical()
    to within numerical precision — use verify_linearisation() to check.

    Returns:
        A (4×4): state transition matrix  (analytical)
        B (4×1): control input matrix     (analytical)
    """
    # Derived coefficients (from equations above)
    a11 = -G * (M1 + M2) / (M1 * L1)   # ∂θ̈1/∂θ1
    a12 =  G * M2 / (M1 * L1)          # ∂θ̈1/∂θ2
    a21 =  G * (M1 + M2) / (M1 * L2)   # ∂θ̈2/∂θ1
    a22 = -G * (M1 + M2) / (M1 * L2)   # ∂θ̈2/∂θ2

    b1  =  1.0 / (2 * M1 * L1)         # ∂θ̈1/∂force_val  (torque → angular acceleration)
    # b2 = 0: force_val acts only on M1, not M2

    # State order: [θ1, θ2, ω1, ω2]
    # Rows 0,1: trivial  (θ̇1=ω1, θ̇2=ω2)
    # Rows 2,3: linearised accelerations from above
    A = np.array([
        [  0,   0,  1,  0],   # dθ1/dt = ω1
        [  0,   0,  0,  1],   # dθ2/dt = ω2
        [a11, a12,  0,  0],   # dω1/dt = θ̈1
        [a21, a22,  0,  0],   # dω2/dt = θ̈2
    ])

    B = np.array([[0], [0], [b1], [0]])

    return A, B

# Verification Helper
def verify_linearisation(tol=1e-4):
    """
    Checks that the analytical and numerical linearisations agree to within
    tolerance. Prints a pass/fail report — call this once at startup.
    """
    A_num, B_num = linearise_numerical()
    A_ana, B_ana = linearise_analytical()

    err_A = np.max(np.abs(A_num - A_ana))
    err_B = np.max(np.abs(B_num - B_ana))

    print("── Linearisation Verification ──────────────────────")
    print(f"  Max |A_numerical − A_analytical| = {err_A:.2e}  ", end="")
    print("✓ PASS" if err_A < tol else "✗ FAIL")
    print(f"  Max |B_numerical − B_analytical| = {err_B:.2e}  ", end="")
    print("✓ PASS" if err_B < tol else "✗ FAIL")
    print("────────────────────────────────────────────────────\n")

    return err_A < tol and err_B < tol