from gettext import find

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import shallow_network as n, plot_util as plu, data_util as dtu
import json_util as jul

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

if __name__ == '__main__':
    input_size = 28 * 28
    output_size = 10
    max_epochs = 5000  #5000
    train_set_len = 10000
    test_set_len = 5000

    hidden_layer_size = [8, 16, 32, 64, 128]
    alpha_GL = [0.5, 1, 2, 3, 5]                    # soglia di early stopping GL
    alpha_PL = [0.5, 0.75, 1, 2, 3]                 # soglia di early stopping PL

    root = './data'  # Root directory where data will be stored
    plots = './plots'  # Directory dove verranno salvati i grafici
    json_dir = './results_json'  # Directory per i json

    gl_results = []
    pl_results = []

    nets = {}
    optimizers = {}
    train_loader = dtu.get_train_loader(root, train_set_len, batch_size=train_set_len)
    test_loader = dtu.get_test_loader(root, test_set_len, batch_size=test_set_len)

    for index in hidden_layer_size:
        # GL Early Stopping
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device)  # questo serve a caricare le reti su gpu se disponibile
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)
        for alpha in alpha_GL:
            train_info = n.get_training_info(train_loader, test_loader, nets[index], device, optimizers[index],
                                             loss_fn=nn.CrossEntropyLoss(), early_stopping=True, alpha=alpha,
                                             use_gl=True, epochs=max_epochs)
            gl_results.append(train_info)
        plu.trace_data(gl_results, "GL", plots)
        jul.save_multiple_train_infos(gl_results, json_dir, "GL")

        # PL Early Stopping
        torch.cuda.empty_cache()
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device)
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)
        for alpha in alpha_PL:
            train_info = n.get_training_info(train_loader, test_loader, nets[index], device, optimizers[index],
                                             loss_fn=nn.CrossEntropyLoss(), early_stopping=True, alpha=alpha,
                                             use_gl=False, epochs=max_epochs)
            pl_results.append(train_info)
        plu.trace_data(pl_results, "PL", plots)
        jul.save_multiple_train_infos(pl_results, json_dir, "PL")

        # No Early Stopping
        torch.cuda.empty_cache()
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device)
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)
        train_info = n.get_training_info(train_loader, test_loader, nets[index], device, optimizers[index],
                                         loss_fn=nn.CrossEntropyLoss(), early_stopping=False, epochs=max_epochs)
        plu.trace_data([train_info], "NoES", plots)
        jul.save_multiple_train_infos([train_info], json_dir, "NoES")

    print(f"[DEBUG] Training complete")
