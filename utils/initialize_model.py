import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from models.texture.cnn import Cnn as TCnn
from models.texture.svm import Svm as TSvm
from models.texture.random_forest import RandomForest as TRandomForest

from models.no_texture.cnn import Cnn
from models.no_texture.svm import Svm
from models.no_texture.random_forest import RandomForest

from utils.load_config import load_config
from utils.preprocess_images import preprocess_images

def evaluate_model_wrapper(model_type, config, X_train, y_train, X_val, y_val):
    texture = config['model']['texture']

    model_classes = {
        'cnn': (Cnn, TCnn),
        'svm': (Svm, TSvm),
        'random_forest': (RandomForest, TRandomForest)
    }

    if model_type not in model_classes:
        return None

    base_model, texture_model = model_classes[model_type]
    model = texture_model(config, X_train, y_train, X_val, y_val) if texture else base_model(config)

    model.train()

    return model.evaluate()


def initialize_model():
    config = load_config()
    model_types = config['model']['types']  # Assume que é uma lista com os tipos de modelos
    texture = config['model'].get('texture', "no_texture")
    X_train, y_train, X_val, y_val = preprocess_images()

    results_list = []

    # Usando ThreadPoolExecutor para avaliar os modelos em paralelo
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_model_wrapper, model_type, config, X_train, y_train, X_val, y_val): model_type for
            model_type in model_types}

        for future in futures:
            model_type = futures[future]
            try:
                result = future.result()
                if result is not None:
                    result['model_type'] = model_type  # Adiciona o tipo do modelo aos resultados
                    results_list.append(result)
            except Exception as e:
                print(f"Ocorreu um erro ao avaliar o modelo {model_type}: {e}")

    # Crie um DataFrame a partir dos resultados
    results_df = pd.DataFrame(results_list)

    # Salve os resultados em um arquivo CSV
    results_df.to_csv('result/{}.csv'.format(texture), index=False)
    print("Resultados salvos em {}.csv".format(texture))

    return
