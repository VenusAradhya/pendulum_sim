# equations_of_motion.py
# shared physics for the LIGO double pendulum simulation

import numpy as np

# physical constants — defined once here so both scripts always use the same values
M1, M2 = 20.0, 20.0  # mirror masses = 20 kg
L1, L2 = 1.0, 1.0    # string lengths = 1 m
G = 9.81              # gravitational acceleration m/s^2

# damping parameters — Q factor of 100 models a lightly damped suspension (realistic for LIGO)
# Q = 100 means the pendulum loses energy slowly, taking ~100 oscillations to decay significantly
# w0 = sqrt(G/L) is the natural frequency of each rod (rad/s)
# damping coefficient b = w0 / Q — small number, as expected for a near-frictionless suspension
Q       = 100.0
w0_1    = np.sqrt(G / L1)   # natural frequency of rod 1 (~3.13 rad/s)
w0_2    = np.sqrt(G / L2)   # natural frequency of rod 2 (~3.13 rad/s)
b1      = w0_1 / Q          # damping coefficient for rod 1 (~0.031)
b2      = w0_2 / Q          # damping coefficient for rod 2 (~0.031)

def equations_of_motion(state, x_p_ddot, force_val):
    th1, th2, w1, w2 = state  #defines state to be a list of [𝜃1, 𝜃2, θ'1, θ'2]
    delta = th1 - th2  #phase difference between two mirrors

    # eq 11: theta1 acceleration
    num1 = -G * (2*M1 + M2) * np.sin(th1)
    num2 = -M2 * G * np.sin(th1 - 2*th2)
    num3 = -2 * np.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(delta))
    den  = (2*M1 + M2 - M2 * np.cos(2*delta))
    # suspension point acceleration projects onto rod 1 as a pseudo-force on both masses
    # equivalent to ground shaking pushing both M1 and M2 sideways - seismic noise enters here
    num_sp1 = -(2*M1 + M2) * x_p_ddot * np.cos(th1)
    # control force on M1 becomes a torque F1*L1*cos(th1), must keep L1 here since next line divides by L1*den
    num_f1  = force_val * L1 * np.cos(th1)
    # viscous damping on rod 1: opposes angular velocity, scaled by damping coefficient b1
    # enters as -b1 * (2M1+M2) * L1 * w1 so it divides cleanly through L1*den to give -b1*(2M1+M2)*w1/den
    num_d1  = -b1 * (2*M1 + M2) * L1 * w1
    th1_acc = (num1 + num2 + num3 + num_sp1 + num_f1 + num_d1) / (L1 * den)

    # eq 12: theta2 acceleration
    num4 = 2 * np.sin(delta)
    num5 = w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(th1) + w2**2 * L2 * M2 * np.cos(delta)
    # suspension point acceleration also loads onto rod 2 via both masses sitting above M2
    num_sp2 = -(M1 + M2) * x_p_ddot * np.cos(th2)
    # viscous damping on rod 2: same form, b2 * M2 * L2 * w2 opposes rotation of lower rod
    num_d2  = -b2 * M2 * L2 * w2
    th2_acc = (num4 * num5 + num_sp2 + num_d2) / (L2 * den)

    #returns an array with velocities and acc of thetas
    return np.array([w1, w2, th1_acc, th2_acc])
