import numpy as np

def euler(F, X, t0, tmax, dt):
    r"""
    Function to apply Euler's Method to solve a 1st Order Differential Equation
    
    F: np.array, array of functions to solve for
    X: list, array of inital consitions for each function
    t0: int, start time
    tmax: int, maximum time
    dt: float, time step
    """
    t = np.arange(t0, tmax + dt, dt)
    x = np.zeros((len(t), len(X)))
    x[0,:] = X
    for n in range(len(t)-1):
        x[n+1,:] = x[n, :] + dt*F(t[n], x[n, :])
    return t, x