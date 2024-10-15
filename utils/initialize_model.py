from models.cnn import Cnn
from models.svm import Svm
from models.random_forest import RandomForest

from utils.load_config import load_config
from utils.preprocess_images import preprocess_images

def initialize_model():
  config = load_config()
  model = config['model']['type']
  X_train, y_train, X_val, y_val = preprocess_images()

  if model == 'cnn':
    cnn = Cnn(config, X_train, y_train, X_val, y_val)
    cnn.train()
    cnn.evaluate()

  elif model == 'svm':
    svm = Svm(config, X_train, y_train, X_val, y_val)
    svm.train()
    svm.evaluate()

  elif model == 'random_forest':
    random_forest = RandomForest(config, X_train, y_train, X_val, y_val)
    random_forest.train()
    random_forest.evaluate()
  
  print("Modelo não existe.")
  return

  