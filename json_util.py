import json
import os
from dataclasses import asdict

def save_train_info_json(train_info, folder, prefix):
    """
    Salva un singolo TrainInfo in formato JSON.
    """
    os.makedirs(folder, exist_ok=True)

    filename = f"{prefix}_hidden{train_info.hidden_layer_size}_alpha{train_info.alpha}.json"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w") as f:
        json.dump(asdict(train_info), f, indent=4)


def save_multiple_train_infos(train_infos, folder, prefix):
    """
    Salva una lista di TrainInfo in JSON separati.
    """
    for idx, train_info in enumerate(train_infos):
        save_train_info_json(train_info, folder, f"{prefix}_{idx}")