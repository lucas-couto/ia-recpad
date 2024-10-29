from keras import layers, models
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class DenseNetwork:
    def __init__(self, config, X_train, y_train, X_val, y_val):
        self.input_shape = tuple(config['model']['input_shape'])
        self.num_classes = config['model']['num_classes']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']

        # Definindo n_features com base no tamanho do vetor de características
        self.n_features = X_train.shape[1]  # Assume que X_train é um array 2D (num_samples, n_features)
        print(self.n_features)

        self.model = self.build_model()

        self.label_map = {label: idx for idx, label in enumerate(set(y_train))}
        y_train = [self.label_map[label] for label in y_train]
        y_val = [self.label_map[label] for label in y_val]

        self.X_train, self.y_train, self.X_val, self.y_val = self.load_data(X_train, y_train, X_val, y_val)

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.n_features,)))  # Use self.n_features
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self, X_train, y_train, X_val, y_val):
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0

        y_train = to_categorical(y_train, self.num_classes)
        y_val = to_categorical(y_val, self.num_classes)

        return X_train, y_train, X_val, y_val

    def train(self):
        history = self.model.fit(self.X_train, self.y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(self.X_val, self.y_val))
        return history

    def evaluate(self):
        # Avalia o modelo e obtém a acurácia
        test_loss, test_acc = self.model.evaluate(self.X_val, self.y_val)

        # Faz previsões
        y_pred_probs = self.model.predict(self.X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Converte y_val de volta para rótulos originais
        y_val_labels = np.argmax(self.y_val, axis=1)

        # Calcula métricas de precisão, recall e F1
        precision = precision_score(y_val_labels, y_pred, average='weighted')
        recall = recall_score(y_val_labels, y_pred, average='weighted')
        f1 = f1_score(y_val_labels, y_pred, average='weighted')

        # Retorna as métricas em um dicionário
        return {
            "accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def predict(self, X):
        return self.model.predict(X)
