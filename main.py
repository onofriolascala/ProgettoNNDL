from gettext import find

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import shallow_network as n, plot_util as plu, data_util as dtu

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_size = 28 * 28
    output_size = 10
    max_epochs = 5000
    batch_size = 10000

    hidden_layer_size = [8, 16, 32, 64, 128]
    alpha_GL = [0.5, 1, 1.5, 2, 2.5]                   # soglia di early stopping GL
    alpha_PL = [0.5, 1, 1.5, 2, 2.5]                   # soglia di early stopping PL

    root = './data'                 # Root directory where data will be stored
    plots = './plots'               # Directory dove verranno salvati i grafici

    gl_results = {}
    pl_results = {}
    std_results = {}
    nets = {}
    optimizers = {}
    train_loader = dtu.get_train_loader(root, 10000, batch_size)
    test_loader = dtu.get_test_loader(root, 5000, batch_size)
    for index in hidden_layer_size:
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device) #questo serve a caricare le reti su gpu se disponibile
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)

    for index in hidden_layer_size:
        # GL Early Stopping
        for alpha in alpha_GL:
            train_loss, val_loss, train_accuracy, val_accuracy, epochs = n.train_net(
                train_loader, test_loader, nets[index], device, optimizers[index],
                True, nn.CrossEntropyLoss(), alpha, True, max_epochs)
            gl_results[index] = [train_loss, val_loss, train_accuracy, val_accuracy, epochs]

        # PL Early Stopping
        for alpha in alpha_PL:
            train_loss, val_loss, train_accuracy, val_accuracy, epochs = n.train_net(
                train_loader, test_loader, nets[index], device, optimizers[index],
                True, nn.CrossEntropyLoss(), alpha, False, max_epochs)
            pl_results[index] = [train_loss, val_loss, train_accuracy, val_accuracy, epochs]

        # Without Early Stopping
        train_loss, val_loss, train_accuracy, val_accuracy, epochs = n.train_net(
                train_loader, test_loader, nets[index], device, optimizers[index],
                False, nn.CrossEntropyLoss(), 1, False, max_epochs)
        std_results[index] = [train_loss, val_loss, train_accuracy, val_accuracy, epochs]

    key = input("Digitare 'plot' per salvare localmente i plot")
    if 'plot' in key:
        for index in hidden_layer_size:
            plu.generic_plot()

    print(f"[DEBUG] Training complete")
