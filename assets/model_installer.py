import os

import requests
from tqdm import tqdm

PREDICTORS = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/predictors/"#"https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/"
EMBEDDERS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"

PREDICTORS_DIR = os.path.join(os.getcwd(), "rvc", "models", "predictors")
EMBEDDERS_DIR = os.path.join(os.getcwd(), "rvc", "models", "embedders")

# Создаем папки, если их нет
os.makedirs(PREDICTORS_DIR, exist_ok=True)
os.makedirs(EMBEDDERS_DIR, exist_ok=True)


def dl_model(link, model_name, dir_name):
    file_path = os.path.join(dir_name, model_name)
    if os.path.exists(file_path):
        return  # Пропускаем загрузку, если файл уже существует

    r = requests.get(f"{link}{model_name}", stream=True)
    r.raise_for_status()

    # Получаем общий размер файла
    total_size = int(r.headers.get("content-length", 0))
    # Используем tqdm для отображения прогресса
    with open(file_path, "wb") as f, tqdm(
        desc=f"Установка {model_name}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def check_and_install_models(f0_onnx=False):
    try:
        predictors_names = ["rmvpe.pt", "fcpe.pt", "fcpe_legacy.pt", "crepe_full.pth", "crepe_large.pth", "crepe_medium.pth", "crepe_small.pth", "crepe_tiny.pth", "fcn.pt"]
        if f0_onnx:
            predictors_names += ["rmvpe.onnx", "fcpe.onnx", "fcpe_legacy.onnx", "crepe_full.onnx", "crepe_large.onnx", "crepe_medium.onnx", "crepe_small.onnx", "crepe_tiny.onnx", "fcn.onnx"]

        for model in predictors_names:
            dl_model(PREDICTORS, model, PREDICTORS_DIR)

        embedder_names = ["hubert_base.pt"]
        for model in embedder_names:
            dl_model(EMBEDDERS, model, EMBEDDERS_DIR)

    except requests.exceptions.RequestException as e:
        print(f"Произошла ошибка при загрузке модели: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
