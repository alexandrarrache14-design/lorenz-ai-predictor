import matplotlib.pyplot as plt
from src.lorenz import generar_datos_lorenz
from src.dataset import PreprocesadorDatos, dividir_entrenamiento_prueba
from src.window import crear_ventanas_temporales
from src.models import crear_modelo_lstm
from src.predict import predecir_cadena_autoregresiva

# 1. Simulación
print("1. Generando trayectoria del Atractor de Lorenz...")
datos_reales = generar_datos_lorenz(pasos=8000)

# 2. Preprocesamiento
preprocesador = PreprocesadorDatos()
datos_norm = preprocesador.normalizar(datos_reales)

X, Y = crear_ventanas_temporales(datos_norm, tamano_ventana=10)
X_train, X_test = dividir_entrenamiento_prueba(X, 0.8)
Y_train, Y_test = dividir_entrenamiento_prueba(Y, 0.8)

# 3. Entrenamiento
print("2. Entrenando la Red Neuronal LSTM...")
modelo = crear_modelo_lstm(tamano_ventana=10)
modelo.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)

# 4. Inferencia
print("3. Realizando predicción autorregresiva...")
PASOS = 400
pred_norm = predecir_cadena_autoregresiva(modelo, X_test[0], pasos_prediccion=PASOS)
pred_reales = preprocesador.desnormalizar(pred_norm)

corte = int(len(X) * 0.8)
real_comparacion = datos_reales[corte + 10 : corte + 10 + PASOS]

# 5. Graficación 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(real_comparacion[:, 0], real_comparacion[:, 1], real_comparacion[:, 2], 
        label='Real (RK4)', color='navy')
ax.plot(pred_reales[:, 0], pred_reales[:, 1], pred_reales[:, 2], 
        label='Predicción IA (LSTM)', color='crimson', linestyle='--')
ax.set_title('Atractor de Lorenz: Real vs Predicción LSTM')
ax.legend()
plt.show()

#Une todo el código en orden y genera la gráfica 3D comparativa.