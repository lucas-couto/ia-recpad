import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image


class RandomForest:
    def __init__(self, config):
        self.n_estimators = config['random_forest']['n_estimators']
        self.max_depth = config['random_forest']['max_depth']
        self.batch_size = config['training']['batch_size']
        self.image_size = config['training'].get('image_size', (128, 128))  # Tamanho padrão das imagens
        self.train_data = self.load_data(config['paths']['train_dir'])
        self.validation_data = self.load_data(config['paths']['valid_dir'])
        self.model = self.build_model()

    def build_model(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)

    def load_data(self, data_dir):
        """Carrega imagens e seus rótulos de subpastas em um diretório."""
        data = []
        labels = []
        # Percorre cada subpasta no diretório
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):  # Verifica se é uma pasta
                for filename in os.listdir(class_path):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        label = class_dir  # Usa o nome da subpasta como rótulo
                        data.append(os.path.join(class_path, filename))
                        labels.append(label)
        return data, labels

    def load_image(self, file_path):
        """Carrega e processa uma imagem."""
        try:
            image = Image.open(file_path)
            image = image.resize(self.image_size)
            return np.array(image).flatten()  # Converte a imagem em um vetor 1D
        except Exception as e:
            print(f"Erro ao carregar a imagem {file_path}: {e}")
            return None  # Retorna None se ocorrer um erro

    def prepare_data(self, data):
        """Prepara os dados de imagem e rótulos para treinamento ou validação."""
        X = []
        for file in data[0]:
            image_vector = self.load_image(file)
            if image_vector is not None:
                X.append(image_vector)

        X = np.array(X)  # Converte a lista de imagens em um array
        y = np.array(data[1])  # Rótulos

        if X.size == 0:
            raise ValueError("Nenhuma imagem válida foi carregada. Verifique os caminhos e formatos.")

        return X, y

    def train(self):
        X_train, y_train = self.prepare_data(self.train_data)
        print("Forma de X_train:", X_train.shape)  # Para depuração
        print("Forma de y_train:", y_train.shape)  # Para depuração
        self.model.fit(X_train, y_train)  # Ajustar o modelo com todos os dados de treinamento

    def evaluate(self):
        X_val, y_val = self.prepare_data(self.validation_data)
        print("Forma de X_val:", X_val.shape)  # Para depuração
        print("Forma de y_val:", y_val.shape)  # Para depuração
        y_pred = self.model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"Acurácia no conjunto de validação: {accuracy * 100:.2f}%")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def predict(self, X):
        return self.model.predict(X)
