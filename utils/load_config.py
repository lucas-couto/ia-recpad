import yaml

# Função para carregar configurações a partir de um arquivo YAML
def load_config():
    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)  # Carrega o YAML de forma segura
    return config