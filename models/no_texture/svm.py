import os
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARNING, 2 = ERROR

class Svm:
    def __init__(self, config):
        self.input_shape = tuple(config['model']['input_shape'])
        self.num_classes = config['model']['num_classes']
        self.train_dir = config['paths']['train_dir']
        self.valid_dir = config['paths']['valid_dir']
        self.batch_size = config['training']['batch_size']
        self.kernel = config['svm']['kernel']
        self.C = config['svm']['C']
        self.gamma = config['svm']['gamma']
        self.train_data, self.validation_data = self.load_data()
        self.model = self.build_model()

    def load_data(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)

        train_data = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        validation_data = valid_datagen.flow_from_directory(
            self.valid_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        return train_data, validation_data

    def build_model(self):
        model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        return model

    def train(self):
        X_train, y_train = self._get_data(self.train_data)
        self.model.fit(X_train, y_train)

    def evaluate(self):
        X_val, y_val = self._get_data(self.validation_data)
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy}")
        return accuracy

    def _get_data(self, data_generator):
        X, y = [], []
        for x_batch, y_batch in data_generator:  # Iterar diretamente sobre o gerador
            X.append(x_batch.reshape(self.batch_size, -1))
            y.append(y_batch)
        return np.vstack(X), np.vstack(y)

    def predict(self, X):
        return self.model.predict(X.reshape(len(X), -1))
