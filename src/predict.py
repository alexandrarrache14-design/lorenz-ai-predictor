import numpy as np

def predecir_cadena_autoregresiva(modelo, secuencia_inicial, pasos_prediccion=400):
    """
    Predice en cadena usando la propia respuesta del modelo como entrada
    para el siguiente instante de tiempo.
    """
    secuencia_actual = secuencia_inicial.copy()
    predicciones = []

    for _ in range(pasos_prediccion):
        entrada = secuencia_actual[np.newaxis, :, :]
        pred = modelo.predict(entrada, verbose=0)[0]
        predicciones.append(pred)
        # Desplazar ventana
        secuencia_actual = np.vstack([secuencia_actual[1:], pred])

    return np.array(predicciones)

#Hace la predicción en cadena. La red usa sus propias respuestas anteriores 
#para adivinar el futuro y ver dónde se desvía por el caos.