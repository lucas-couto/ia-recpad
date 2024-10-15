from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForest():
    def __init__(self, config, X_train, y_train, X_val, y_val):
        self.n_estimators = config['model']['n_estimators']
        self.max_depth = config['model']['max_depth']
        self.model = self.build_model()
        self.X_train, self.y_train, self.X_val, self.y_val = self.load_data(X_train, y_train, X_val, y_val)

    def build_model(self):
        # Criar o modelo Random Forest
        model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
        return model

    def load_data(self, X_train, y_train, X_val, y_val):
        # Aqui podemos simplesmente retornar os dados, sem normalização
        return X_train, y_train, X_val, y_val

    def train(self):
        # Treinar o modelo Random Forest
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Avaliar o modelo no conjunto de validação
        y_pred = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f"Acurácia no conjunto de validação: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, X):
        # Fazer previsões em novos dados
        return self.model.predict(X)
