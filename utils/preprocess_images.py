import cv2
import os
import numpy as np

from utils.textures import choose_filter
from utils.load_config import load_config

def preprocess_images():
  X_train, y_train = load_images_and_extract_features("datasets/train")
  X_val, y_val = load_images_and_extract_features("datasets/valid")
    
  return X_train, y_train, X_val, y_val



def load_images_and_extract_features(directory):
  config = load_config()

  filter = config['model']['texture']
  features = []
  labels = []

  for folder_name in os.listdir(directory):
      folder_path = os.path.join(directory, folder_name)
      if not os.path.isdir(folder_path):
          continue
      for file_name in os.listdir(folder_path):
          img_path = os.path.join(folder_path, file_name)
          img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
          if img is not None:
              # Extrai as características de textura
              texture_features = choose_filter(filter, img)
              features.append(texture_features)
              labels.append(folder_name)  # Nome da pasta como label
              
  return np.array(features), np.array(labels)