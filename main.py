import os

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARNING, 2 = ERROR
warnings.filterwarnings('ignore')

from utils.organize_files import organize_files
from utils.initialize_model import initialize_model


# Executa a função principal
if __name__ == "__main__":
    initialize_model()