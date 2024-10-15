from keras import layers, models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Cnn():
    def __init__(self, config, X_train, y_train, X_val, y_val):
        self.input_shape = tuple(config['model']['input_shape'])
        self.num_classes = config['model']['num_classes']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.model = self.build_model()
        self.X_train, self.y_train, self.X_val, self.y_val = self.load_data(X_train, y_train, X_val, y_val)

    def build_model(self):
        # Definindo a arquitetura da CNN
        model = models.Sequential()
        
        # Camadas de Convolução e MaxPooling
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Camada Flatten para transformar as features em uma dimensão
        model.add(layers.Flatten())

        # Camadas totalmente conectadas
        model.add(layers.Dense(512, activation='relu'))

        # Camada de saída com o número de classes e ativação softmax
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compilando o modelo
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

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
        test_loss, test_acc = self.model.evaluate(self.X_val, self.y_val)
        print(f"Validation Accuracy: {test_acc}")
        return test_loss, test_acc

    def predict(self, X):
        return self.model.predict(X)
