import numpy as np

class PreprocesadorDatos:
    def __init__(self):
        self.val_min = None
        self.val_max = None

    def normalizar(self, datos):
        """Ajusta los datos al rango [0, 1]."""
        self.val_min = datos.min(axis=0)
        self.val_max = datos.max(axis=0)
        return (datos - self.val_min) / (self.val_max - self.val_min)

    def desnormalizar(self, datos_norm):
        """Devuelve los datos a su escala original."""
        return datos_norm * (self.val_max - self.val_min) + self.val_min

def dividir_entrenamiento_prueba(datos, porcentaje_train=0.8):
    """Divide la serie temporal respetando el orden cronológico."""
    corte = int(len(datos) * porcentaje_train)
    return datos[:corte], datos[corte:]

#Prepara los datos. Escalamos los números entre 0 y 1
#(normalización) y los dividimos en entrenamiento y prueba.