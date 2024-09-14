import numpy as np

def euler(F, X, t0, tmax, dt):
    t = np.arange(t0, tmax + dt, dt)
    x = np.zeros((len(t), len(X)))
    x[0,:] = X
    for n in range(len(t)-1):
        x[n+1,:] = x[n, :] + dt*F(t[n], x[n, :])
    return t, x