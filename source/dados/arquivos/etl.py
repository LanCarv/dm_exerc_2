import zipfile
import os
from dotenv import load_dotenv

load_dotenv()

zip_path = os.getenv("ZIP_PATH_2")
extract_path = os.getenv("EXTRACT_PATH_2")

# Valida os caminhos
if not zip_path or not os.path.exists(zip_path):
    print(f" Arquivo ZIP não encontrado em:\n{zip_path}")
elif not extract_path or not os.path.exists(extract_path):
    print(f"Pasta de destino não encontrada:\n{extract_path}")
else:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print(f"Arquivos extraídos com sucesso em:\n{extract_path}")