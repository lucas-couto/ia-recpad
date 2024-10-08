from models.cnn import Cnn
from utils.load_config import load_config

def initialize_model():
    config = load_config()

    if config['model']['type'] == 'cnn':
      cnn = Cnn(config)
      cnn.train(config['training']['epochs'], config['training']['batch_size'])
      cnn.evaluate()