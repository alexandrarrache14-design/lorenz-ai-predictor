import numpy as np

def lorenz_derivadas(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Calcula las derivadas del sistema de Lorenz."""
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def generar_datos_lorenz(pasos=8000, dt=0.01, estado_inicial=(1.0, 1.0, 1.0)):
    """Resuelve el sistema de Lorenz usando Runge-Kutta de 4to Orden (RK4)."""
    puntos = np.zeros((pasos, 3))
    puntos[0] = estado_inicial

    for i in range(pasos - 1):
        x, y, z = puntos[i]
        k1 = lorenz_derivadas(x, y, z)
        k2 = lorenz_derivadas(*(puntos[i] + 0.5 * dt * k1))
        k3 = lorenz_derivadas(*(puntos[i] + 0.5 * dt * k2))
        k4 = lorenz_derivadas(*(puntos[i] + dt * k3))
        puntos[i+1] = puntos[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return puntos

#el motor matemático. Usa el algoritmo RK4
#para calcular la trayectoria real de la mariposa (los datos exactos).