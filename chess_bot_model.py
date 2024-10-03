import tensorflow as tf
from tensorflow.keras import layers


def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(64,)),  # L'échiquier est représenté par 64 cases
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='linear')  # 64 sorties pour les coups possibles
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
