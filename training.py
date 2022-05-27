import tensorflow as tf
import numpy as np
from config import CONFIG
from utility_functions import windowed_dataset
from data_preparation import get_data

tf.random.set_seed(51)
np.random.seed(51)

def create_and_model():
    " Create a LSTM model and training"
    x_train = get_data(train = True)
    train_set = windowed_dataset(x_train, CONFIG.window_size=60, batch_size=CONFIG.BATCH_SIZE, shuffle_buffer=CONFIG.shuffle_buffer_size)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(200, return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(100, return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer="adam",
                  metrics=["mae"])
    history = model.fit(train_set,epochs=CONFIG.EPOCHS)
    return model
