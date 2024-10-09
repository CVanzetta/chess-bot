import tensorflow as tf
from tensorflow import keras

def create_model():
    # Total number of possible moves (64 starting squares * 64 destination squares * 6 possible promotions)
    N = 64 * 64 * 6  # 24,576
    model = keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(N, activation='linear')  # N outputs for all possible moves
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model