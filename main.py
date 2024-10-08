from utils.organize_files import organize_files
from utils.initialize_model import initialize_model

def main():
  organize_files("datasets/train")
  organize_files("datasets/valid")
  initialize_model()

main()