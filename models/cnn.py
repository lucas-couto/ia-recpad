from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator

class Cnn():
    def __init__(self, config):
        self.input_shape = tuple(config['model']['input_shape'])
        self.num_classes = config['model']['num_classes']
        self.train_dir = config['paths']['train_dir']
        self.valid_dir = config['paths']['valid_dir']
        self.batch_size = config['training']['batch_size']
        self.model = self.build_model()
        self.train_data, self.validation_data = self.load_data()

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

    def load_data(self):
        # Geradores de imagens com augmentação para o conjunto de treino
        train_datagen = ImageDataGenerator(rescale=1./255,   # Normalizar as imagens
                                           shear_range=0.2,  # Aplicar transformações de corte
                                           zoom_range=0.2,   # Aplicar zoom
                                           horizontal_flip=True)  # Flip horizontal

        # Gerador de imagens para o conjunto de validação (sem augmentação)
        valid_datagen = ImageDataGenerator(rescale=1./255)  # Apenas normalizar as imagens de validação

        # Carregar os dados de treino e validação
        train_data = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),  # Redimensionar todas as imagens
            batch_size=self.batch_size,
            class_mode='categorical')  # Modo categórico para classificação multiclasse

        validation_data = valid_datagen.flow_from_directory(
            self.valid_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical')

        return train_data, validation_data

    def train(self, epochs=10, batch_size=32):
        # Treinamento do modelo
        history = self.model.fit(self.train_data, 
                                 epochs=epochs, 
                                 batch_size=batch_size,
                                 validation_data=self.validation_data)
        return history

    def evaluate(self):
        # Avaliação no conjunto de validação
        test_loss, test_acc = self.model.evaluate(self.validation_data)
        print(f"Validation Accuracy: {test_acc}")
        return test_loss, test_acc

    def predict(self, X):
        # Fazendo previsões
        return self.model.predict(X)