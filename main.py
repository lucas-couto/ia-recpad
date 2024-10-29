import os
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARNING, 2 = ERROR
warnings.filterwarnings('ignore')

from utils.initialize_model import initialize_model


import tensorflow as tf
tf.keras.backend.clear_session()


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Executa a função principal
# if __name__ == "__main__":
    # initialize_model()