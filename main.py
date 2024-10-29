import os
import warnings
from utils.initialize_model import initialize_model

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')
    initialize_model()

if __name__ == "__main__":
    main()