from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Svm():
    def __init__(self, config, X_train, y_train, X_val, y_val):
        self.kernel = config['model']['kernel']
        self.C = config['model']['C']
        self.gamma = config['model']['gamma']
        self.scaler = StandardScaler()  # Normalização dos dados
        self.model = self.build_model()
        self.X_train, self.y_train, self.X_val, self.y_val = self.load_data(X_train, y_train, X_val, y_val)

    def build_model(self):
        # Criar o modelo SVM
        model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        return model

    def load_data(self, X_train, y_train, X_val, y_val):
        # Normalizar os dados de treino e validação
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        return X_train, y_train, X_val, y_val

    def train(self):
        # Treinar o modelo SVM
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Avaliar o modelo no conjunto de validação
        y_pred = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f"Acurácia no conjunto de validação: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, X):
        # Fazer previsões em novos dados
        X = self.scaler.transform(X)
        return self.model.predict(X)
