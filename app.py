import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.lorenz import generar_datos_lorenz
from src.dataset import PreprocesadorDatos, dividir_entrenamiento_prueba
from src.window import crear_ventanas_temporales
from src.models import crear_modelo_ffn, crear_modelo_rnn, crear_modelo_lstm
from src.predict import predecir_cadena_autoregresiva

# Configuración de la página
st.set_page_config(page_title="Lorenz AI Predictor", page_icon="🦋", layout="wide")

# Colores y estilos sencillos para que se vea llamativa
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1 {
        color: #38bdf8;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦋 Lorenz AI Predictor: Física (RK4) vs Inteligencia Artificial")
st.write("Proyecto de Estructuras de Datos - UIS | Comparación de redes neuronales en sistemas caóticos.")

# Panel lateral (Menú de la izquierda)
st.sidebar.header("⚙️ Configuración del Experimento")

tipo_red = st.sidebar.selectbox(
    "Elige la Red Neuronal a probar:",
    ["LSTM (Memoria Larga/Corta)", "RNN (Recurrente Básica)", "FFN (Red Simple sin Memoria)"]
)

pasos_totales = st.sidebar.slider("Puntos a simular (RK4)", 2000, 10000, 5000, 1000)
pasos_prediccion = st.sidebar.slider("Pasos a predecir en el futuro", 100, 800, 300, 50)
epocas = st.sidebar.slider("Épocas de entrenamiento", 5, 30, 10, 5)

# Botón principal
if st.button("🚀 Ejecutar Simulación y Entrenar IA"):
    with st.spinner("Generando datos con RK4 y entrenando la red neuronal..."):
        # 1. Generar trayectoria real
        datos_reales = generar_datos_lorenz(pasos=pasos_totales)
        
        # 2. Normalizar y crear ventanas de 10 pasos
        preprocesador = PreprocesadorDatos()
        datos_norm = preprocesador.normalizar(datos_reales)
        
        X, Y = crear_ventanas_temporales(datos_norm, tamano_ventana=10)
        X_train, X_test = dividir_entrenamiento_prueba(X, 0.8)
        Y_train, Y_test = dividir_entrenamiento_prueba(Y, 0.8)
        
        # 3. Elegir el modelo según lo que seleccionó el usuario
        if "LSTM" in tipo_red:
            modelo = crear_modelo_lstm(tamano_ventana=10)
        elif "RNN" in tipo_red:
            modelo = crear_modelo_rnn(tamano_ventana=10)
        else:
            modelo = crear_modelo_ffn(tamano_ventana=10)
            
        modelo.fit(X_train, Y_train, epochs=epocas, batch_size=32, verbose=0)
        
        # 4. Predecir hacia el futuro
        pred_norm = predecir_cadena_autoregresiva(modelo, X_test[0], pasos_prediccion=pasos_prediccion)
        pred_reales = preprocesador.desnormalizar(pred_norm)
        
        corte = int(len(X) * 0.8)
        real_comp = datos_reales[corte + 10 : corte + 10 + pasos_prediccion]
        
        # 5. Calcular errores MAE y RMSE (como dice el informe)
        mae = np.mean(np.abs(real_comp - pred_reales))
        rmse = np.sqrt(np.mean((real_comp - pred_reales) ** 2))
        
    st.success(f"¡Simulación completada usando el modelo {tipo_red}!")
    
    # Mostrar métricas en 3 columnas bonitas
    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo Usado", tipo_red.split()[0])
    col2.metric("Error MAE", f"{mae:.4f}")
    col3.metric("Error RMSE", f"{rmse:.4f}")
    
    # Botón para exportar el dataset sintético (Pendiente del informe)
    df_datos = pd.DataFrame(datos_reales, columns=["X", "Y", "Z"])
    csv = df_datos.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Dataset Simulado (CSV)",
        data=csv,
        file_name="dataset_lorenz_rk4.csv",
        mime="text/csv"
    )

    # 6. Gráfica 3D con fondo oscuro y colores vivos
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0e1117')
    
    ax.plot(real_comp[:, 0], real_comp[:, 1], real_comp[:, 2], label='Real (RK4)', color='#38bdf8', linewidth=2)
    ax.plot(pred_reales[:, 0], pred_reales[:, 1], pred_reales[:, 2], label=f'IA ({tipo_red.split()[0]})', color='#f43f5e', linestyle='--', linewidth=2)
    
    ax.set_title("Atractor de Lorenz: Real (Azul) vs Predicción IA (Rojo)", color='white')
    ax.set_xlabel("Eje X", color='white')
    ax.set_ylabel("Eje Y", color='white')
    ax.set_zlabel("Eje Z", color='white')
    ax.tick_params(colors='white')
    ax.legend()
    
    st.pyplot(fig)
