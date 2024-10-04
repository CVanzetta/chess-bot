import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    # Nombre total de coups possibles (64 cases de départ * 64 cases d'arrivée * 6 promotions possibles)
    N = 64 * 64 * 6  # 24 576
    model = tf.keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(N, activation='linear')  # N sorties pour tous les coups possibles
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
