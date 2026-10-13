import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.lorenz import generar_datos_lorenz
from src.dataset import PreprocesadorDatos, dividir_entrenamiento_prueba
from src.window import crear_ventanas_temporales
from src.models import crear_modelo_lstm
from src.predict import predecir_cadena_autoregresiva

# Configuración de la ventana en el navegador
st.set_page_config(page_title="Simulador Atractor de Lorenz & IA", layout="wide")

st.title("Atractor de Lorenz: Simulación Matemática vs Predicción IA")
st.write("Esta aplicación permite simular la trayectoria del Atractor de Lorenz mediante integración numérica (RK4) y evaluar cómo la Red Neuronal LSTM predice el comportamiento caótico.")

# Panel lateral con opciones
st.sidebar.header("Parámetros del Experimento")
pasos_totales = st.sidebar.slider("Puntos a simular (RK4)", min_value=2000, max_value=10000, value=5000, step=1000)
pasos_prediccion = st.sidebar.slider("Pasos a predecir con IA", min_value=100, max_value=800, value=300, step=50)
epocas = st.sidebar.slider("Épocas de entrenamiento", min_value=5, max_value=30, value=10, step=5)

# Botón principal
if st.button("Ejecutar Simulación y Entrenar IA"):
    with st.spinner("Generando datos reales con RK4 y entrenando la Red Neuronal..."):
        # 1. Simulación matemática
        datos_reales = generar_datos_lorenz(pasos=pasos_totales)
        
        # 2. Preprocesamiento y normalización
        preprocesador = PreprocesadorDatos()
        datos_norm = preprocesador.normalizar(datos_reales)
        
        X, Y = crear_ventanas_temporales(datos_norm, tamano_ventana=10)
        X_train, X_test = dividir_entrenamiento_prueba(X, 0.8)
        Y_train, Y_test = dividir_entrenamiento_prueba(Y, 0.8)
        
        # 3. Entrenamiento del modelo LSTM
        modelo = crear_modelo_lstm(tamano_ventana=10)
        modelo.fit(X_train, Y_train, epochs=epocas, batch_size=32, verbose=0)
        
        # 4. Predicción autorregresiva
        pred_norm = predecir_cadena_autoregresiva(modelo, X_test[0], pasos_prediccion=pasos_prediccion)
        pred_reales = preprocesador.desnormalizar(pred_norm)
        
        corte = int(len(X) * 0.8)
        real_comp = datos_reales[corte + 10 : corte + 10 + pasos_prediccion]
        
    st.success("¡Simulación y predicción completadas con éxito!")

    # 5. Gráfica 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(real_comp[:, 0], real_comp[:, 1], real_comp[:, 2], label='Real (RK4)', color='navy', linewidth=1.5)
    ax.plot(pred_reales[:, 0], pred_reales[:, 1], pred_reales[:, 2], label='Predicción IA (LSTM)', color='crimson', linestyle='--', linewidth=1.5)
    
    ax.set_title("Atractor de Lorenz: Real (Azul) vs Predicho por IA (Rojo)")
    ax.set_xlabel("Eje X")
    ax.set_ylabel("Eje Y")
    ax.set_zlabel("Eje Z")
    ax.legend()
    
    st.pyplot(fig)