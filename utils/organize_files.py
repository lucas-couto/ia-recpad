import os
import shutil
import pandas as pd

def organize_files(path):
  image_folder = path

  csv_file = '{}/_classes.csv'.format(path)
  df = pd.read_csv(csv_file)

  fresh_folder = '{}/fresh'.format(path)
  spoiled_folder = '{}/spoiled'.format(path)
  half_fresh_folder = '{}/half-fresh'.format(path)

  os.makedirs(fresh_folder, exist_ok=True)
  os.makedirs(half_fresh_folder, exist_ok=True)
  os.makedirs(spoiled_folder, exist_ok=True)

  for index, row in df.iterrows():
      filename = row['filename']
      src_path = os.path.join(image_folder, filename)

      if row['Fresh'] == 1:
          dest_path = os.path.join(fresh_folder, filename)
      elif row['Half-Fresh'] == 1:
          dest_path = os.path.join(half_fresh_folder, filename)
      elif row['Spoiled'] == 1:
          dest_path = os.path.join(spoiled_folder, filename)
      
      if os.path.exists(src_path):
          shutil.move(src_path, dest_path)
          print(f'Movido: {filename} para {dest_path}')
      else:
          print(f'Arquivo não encontrado: {filename}')
