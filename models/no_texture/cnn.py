from keras import layers, models
import os
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARNING, 2 = ERROR

class Cnn:
    def __init__(self, config):
        self.input_shape = tuple(config['model']['input_shape'])
        self.num_classes = config['model']['num_classes']
        self.train_dir = config['paths']['train_dir']
        self.valid_dir = config['paths']['valid_dir']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.model = self.build_model()
        self.train_data, self.validation_data = self.load_data()

    def build_model(self):
        # Definindo a arquitetura da CNN
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compilando o modelo
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def load_data(self):
        # Geradores de imagens com augmentação para o conjunto de treino
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        # Gerador de imagens para o conjunto de validação (sem augmentação)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)

        # Carregar os dados de treino e validação
        train_data = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical')

        validation_data = valid_datagen.flow_from_directory(
            self.valid_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical')

        return train_data, validation_data

    def train(self):
        # Treinamento do modelo
        history = self.model.fit(self.train_data,
                                 epochs=self.epochs,
                                 validation_data=self.validation_data)
        return history

    def evaluate(self):
        # Avaliação no conjunto de validação
        test_loss, test_acc = self.model.evaluate(self.validation_data)
        print(f"Validation Accuracy: {test_acc}")

        # Faz previsões
        y_pred_probs = self.model.predict(self.validation_data)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Obtém os rótulos verdadeiros
        y_true = self.validation_data.classes  # Rótulos verdadeiros
        y_true = y_true[:len(y_pred)]  # Certifica-se de que os tamanhos sejam compatíveis

        # Calcula as métricas
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Retorna as métricas em um dicionário
        return {
            "accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def predict(self, X):
        # Fazendo previsões
        return self.model.predict(X)
