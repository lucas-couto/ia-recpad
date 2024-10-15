import cv2
import os
import numpy as np

from utils.textures import process_single_image
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
          texture_features = process_single_image(img_path, filter)
          features.append(texture_features)
          labels.append(folder_name)
              
  return np.array(features), np.array(labels)