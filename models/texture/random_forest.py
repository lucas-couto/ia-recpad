from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RandomForest:
    def __init__(self, config, X_train, y_train, X_val, y_val):
        self.n_estimators = config['random_forest']['n_estimators']
        self.max_depth = config['random_forest']['max_depth']
        self.model = self.build_model()
        self.X_train, self.y_train, self.X_val, self.y_val = self.load_data(X_train, y_train, X_val, y_val)

    def build_model(self):
        model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
        return model

    def load_data(self, X_train, y_train, X_val, y_val):
        return X_train, y_train, X_val, y_val

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_val)

        accuracy = accuracy_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred, average='weighted')
        recall = recall_score(self.y_val, y_pred, average='weighted')
        f1 = f1_score(self.y_val, y_pred, average='weighted')

        print(f"Acurácia no conjunto de validação: {accuracy * 100:.2f}%")
        print(f"Precisão: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def predict(self, X):
        return self.model.predict(X)
