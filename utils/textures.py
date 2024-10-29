import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def process_single_image(image_path, filter_name):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (416, 416))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image if filter_name is None else gray_image
    
    feature_vector = choose_filter(filter_name, image)
    return feature_vector

def choose_filter(filterName, image):
  if filterName == "gabor": return gabor_features(image)
  elif filterName == "lbp": return lbp_features(image)
  elif filterName == "glcm": return glcm_features(image)
  elif filterName == "all": 
    gabor_feature_vector, lbp_feature_vector,  glcm_feature_vector = gabor_features(image), lbp_features(image), glcm_features(image)
    return np.hstack([gabor_feature_vector, lbp_feature_vector, glcm_feature_vector])
  
  elif filterName is None:
    return normal_image(image)
  
  return image


# Filtro Gabor
def gabor_features(image):
    gabor_kernels = []
    for theta in range(4):  # Variação angular
        theta = theta / 4. * np.pi
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)
    
    features = [cv2.filter2D(image, cv2.CV_8UC3, kernel).mean() for kernel in gabor_kernels]
    return np.array(features)

# LBP
def lbp_features(image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalização
    return hist

# GLCM
def glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, homogeneity]

def normal_image(image):
   return np.array(image).flatten()