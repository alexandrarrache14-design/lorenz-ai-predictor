import numpy as np

def crear_ventanas_temporales(datos, tamano_ventana=10):
    """
    Convierte una serie de tiempo en pares de (Entrada X, Salida Y).
    X: secuencia de `tamano_ventana` puntos pasados.
    Y: punto inmediatamente siguiente.
    """
    X, Y = [], []
    for i in range(len(datos) - tamano_ventana):
        X.append(datos[i : i + tamano_ventana])
        Y.append(datos[i + tamano_ventana])
    return np.array(X), np.array(Y)

#Agrupa los datos en bloques de 10 puntos pasados 
#para que la red intente adivinar el punto número 11.