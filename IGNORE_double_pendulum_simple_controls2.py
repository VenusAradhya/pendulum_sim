'''
In progress, refer to double_pendulum_simple_controls instead 

Installations
- pip install control

'''
import numpy as np
import control
import matplotlib.pyplot as plt
from double_pendulum_rl import LIGOPendulumEnv, M1, M2, L1, L2, G #pulling physical constants, environment, and masses

def run_lqr_control():
    env = LIGOPendulumEnv() #setting up same env
    
    # linearization with small angle approximation
    # define or A and B matrices: dx/dt = Ax + Bu
    # state x = [th1, th2, w1, w2]
    
    # matrices  derived from the physics equations near the equilibrium (center)
    mu = 1 + M1/M2 #mass ratio of the system
    
    A = np.zeros((4, 4)) # matrix descirbing pendulum swing due to gravity, each row and column: [th1, th2, w1, w2], how it affects each other 
    A[0, 2] = 1.0 #rate of change of pos = vel
    A[1, 3] = 1.0 # ""
    A[2, 0] = -(mu * G) / L1 #restoring forces
    A[2, 1] = (G) / L1 #""
    A[3, 0] = (mu * G) / L2 #""
    A[3, 1] = -(mu * G) / L2 # ""

    B = np.zeros((4, 1)) # matrix descirbing force applied on system at
    B[2, 0] = 1.0 / (M1 * L1) # f = ma applied on pendulum
    B[3, 0] = -1.0 / (M1 * L2) #reaction force

    # lqr optimization
    # questions: how much we care about the state (Mirror positions)
    # result: how much we care about the control force (Energy cost)
    Q = np.diag([100.0, 1000.0, 10.0, 10.0]) # highly penalize bottom mirror (index 1)
    R = np.array([[0.01]]) # cost of force

    # riccati equation to find K
    K, S, E = control.lqr(A, B, Q, R)
    print(f"Optimal LQR Gains calculated: \n{K}")

    # test controller
    obs, _ = env.reset()
    history_x2 = []
    
    for _ in range(1000):
        # classical control Law: u = -K * x
        # note: K is (1,4) and obs is (4,), result is a single force value
        u = -(K @ obs) 
        
        obs, reward, done, _, _ = env.step(u)
        
        # track bottom mirror position
        x2 = L1 * np.sin(obs[0]) + L2 * np.sin(obs[1])
        history_x2.append(x2)
        
        if done: break

    # plot results
    plt.figure(figsize=(10, 5))
    plt.plot(history_x2, label="LQR Stabilization")
    plt.axhline(0, color='red', linestyle='--')
    plt.title("LQR Control: Stabilizing LIGO Bottom Mirror")
    plt.xlabel("Time Steps")
    plt.ylabel("Displacement (m)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_lqr_control()
