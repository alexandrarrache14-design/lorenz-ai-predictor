from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

def crear_modelo_lstm(tamano_ventana=10, num_caracteristicas=3):
    """Crea un modelo recurrente LSTM."""
    modelo = Sequential([
        LSTM(32, input_shape=(tamano_ventana, num_caracteristicas)),
        Dense(16, activation='relu'),
        Dense(num_caracteristicas)
    ])
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

def crear_modelo_ffn(tamano_ventana=10, num_caracteristicas=3):
    """Crea una red densa Feed-Forward estándar."""
    modelo = Sequential([
        Dense(32, activation='relu', input_shape=(tamano_ventana * num_caracteristicas,)),
        Dense(16, activation='relu'),
        Dense(num_caracteristicas)
    ])
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

#Define la arquitectura de la IA. Usamos una red LSTM de 
#32 neuronas especializada en aprender secuencias con memoria.