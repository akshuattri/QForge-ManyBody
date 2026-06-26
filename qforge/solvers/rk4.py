"""RK4 integrator for time-dependent systems."""
import numpy as np

def rk4_step(y: np.ndarray, t: float, dt: float, dydt_func) -> np.ndarray:
    """Single RK4 step."""
    k1 = dydt_func(y, t)
    k2 = dydt_func(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = dydt_func(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = dydt_func(y + dt*k3, t + dt)
    
    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_integrate(y0: np.ndarray, times: np.ndarray, dydt_func) -> np.ndarray:
    """RK4 integration."""
    y = np.zeros((len(times), len(y0)), dtype=y0.dtype)
    y[0] = y0
    
    for i in range(len(times)-1):
        dt = times[i+1] - times[i]
        y[i+1] = rk4_step(y[i], times[i], dt, dydt_func)
    
    return y
