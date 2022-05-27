import matplotlib.pyplot as plt
import tensorflow as tf

from config import CONFIG
from utility_functions import model_forecast
from training import create_and_model
from data_preparation import get_data


model = create_and_model()
x_test,split_time,series = get_data(train = False)
prediction = model_forecast(model, series[..., np.newaxis], CONFIG.window_size)
prediction = prediction[split_time - CONFIG.window_size:-1, -1, 0]

plt.plot(x_test, label='Actual Temperature', color='green')
plt.plot(prediction, label='Predicted Temperature', color='red')
plt.legend(loc='best')
plt.xlabel('Day')
plt.ylabel('Mean Temperature')
plt.show()

print("mae value is :")
tf.keras.metrics.mean_absolute_error(x_test, prediction).numpy()
