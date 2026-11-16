from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Flatten

# 1. Red Simple sin memoria (FFN)
def crear_modelo_ffn(tamano_ventana=10):
    modelo = Sequential([
        Flatten(input_shape=(tamano_ventana, 3)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# 2. Red Recurrente básica (RNN)
def crear_modelo_rnn(tamano_ventana=10):
    modelo = Sequential([
        SimpleRNN(32, activation='tanh', input_shape=(tamano_ventana, 3)),
        Dense(3)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

# 3. Red con compuertas de memoria (LSTM)
def crear_modelo_lstm(tamano_ventana=10):
    modelo = Sequential([
        LSTM(32, activation='tanh', input_shape=(tamano_ventana, 3)),
        Dense(3)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

#Define la arquitectura de la IA. Usamos una red LSTM de 
#32 neuronas especializada en aprender secuencias con memoria.
