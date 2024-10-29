import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from models.dense_network import DenseNetwork
from models.svm import Svm
from models.random_forest import RandomForest

from utils.load_config import load_config
from utils.preprocess_images import preprocess_images

def evaluate_model_wrapper(model_type, config, X_train, y_train, X_val, y_val):
    model_classes = {
        'dense_network': DenseNetwork,
        'svm': Svm,
        'random_forest': RandomForest
    }


    base_model = model_classes[model_type]
    model = base_model(config, X_train, y_train, X_val, y_val)

    model.train()

    return model.evaluate()


def initialize_model():
    config = load_config()
    model_types = config['model']['types']
    texture = config['model'].get('texture', "no_texture")
    X_train, y_train, X_val, y_val = preprocess_images()

    results_list = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_model_wrapper, model_type, config, X_train, y_train, X_val, y_val): model_type for
            model_type in model_types
        }

        for future in futures:
            model_type = futures[future]
            try:
                result = future.result()
                if result is not None:
                    result['model_type'] = model_type
                    results_list.append(result)
            except Exception as e:
                print(f"Ocorreu um erro ao avaliar o modelo {model_type}: {e}")
                traceback.print_exc() 

    # Crie um DataFrame a partir dos resultados
    results_df = pd.DataFrame(results_list)

    # Salve os resultados em um arquivo CSV
    results_df.to_csv('result/{}.csv'.format(texture), index=False)
    print("Resultados salvos em {}.csv".format(texture))

    return
