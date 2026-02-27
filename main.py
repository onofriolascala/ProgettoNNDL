from gettext import find

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import shallow_network as n, plot_util as plu, data_util as dtu

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

if __name__ == '__main__':
    input_size = 28 * 28
    output_size = 10
    max_epochs = 5000
    train_set_len= 10000
    test_set_len= 5000

    hidden_layer_size = [8]#[8, 16, 32, 64, 128]
    alpha_GL = [1]#[0.5, 1, 2, 3, 5]                   # soglia di early stopping GL
    alpha_PL = [1]#[0.5, 0.75, 1, 2, 3]           # soglia di early stopping PL

    root = './data'                 # Root directory where data will be stored
    plots = './plots'               # Directory dove verranno salvati i grafici

    gl_results = []
    pl_results = []
    std_results = []
    nets = {}
    optimizers = {}
    train_loader = dtu.get_train_loader(root, 10000, batch_size)
    test_loader = dtu.get_test_loader(root, 5000, batch_size)
    for index in hidden_layer_size:
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device) #questo serve a caricare le reti su gpu se disponibile
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)

    for index in hidden_layer_size:
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device)  # questo serve a caricare le reti su gpu se disponibile
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)
        # GL Early Stopping
        for alpha in alpha_GL:
            train_info = n.get_training_info(train_loader, test_loader, nets[index], device, optimizers[index],
                                            loss_fn = nn.CrossEntropyLoss(), early_stopping = True, alpha = alpha,
                                            use_gl = True, epochs = max_epochs)
            gl_results.append(train_info)
        torch.cuda.empty_cache()
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device)  # questo serve a caricare le reti su gpu se disponibile
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)
        # PL Early Stopping
        for alpha in alpha_PL:
            train_info = n.get_training_info(train_loader, test_loader, nets[index], device, optimizers[index],
                                            loss_fn = nn.CrossEntropyLoss(), early_stopping = True, alpha = alpha,
                                            use_gl = False, epochs = max_epochs)
            pl_results.append(train_info)

        torch.cuda.empty_cache()
        nets[index] = n.Net(input_size, index, output_size)
        nets[index].to(device)  # questo serve a caricare le reti su gpu se disponibile
        optimizers[index] = torch.optim.Rprop(nets[index].parameters(), lr=0.01)
        # Without Early Stopping
        train_info = n.get_training_info(train_loader, test_loader, nets[index], device, optimizers[index],
                                         loss_fn = nn.CrossEntropyLoss(), early_stopping = False, epochs = max_epochs)
        std_results.append(train_info)

    key = input("Salvare localmente i plot? (y/n)")
    if 'y' in key:
        # GL Plotting
        for train_info in gl_results:
            # Plot Loss/Epochs
            plu.generic_plot(f'GL Loss per nodi interni={train_info.hidden_layer_size} e alpha={train_info.alpha}', "Epochs", "Loss",
                             f'{plots}/gl_loss_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                                [range(1,train_info.max_epoch+2), train_info.train_loss, "Training Loss"],
                                         [range(1,train_info.max_epoch+2), train_info.val_loss, "Validation Loss"])
            # Plot Accuracy/Epochs
            plu.generic_plot(f'GL Accuracy per nodi interni={train_info.hidden_layer_size} e alpha={train_info.alpha}', "Epochs", "Accuracy",
                             f'{plots}/gl_accuracy_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                                [range(1, train_info.max_epoch + 2), train_info.train_accuracy, "Training Accuracy"],
                                         [range(1, train_info.max_epoch + 2), train_info.val_accuracy, "Validation Accuracy"])
        # PL Plotting
        for train_info in pl_results:
            # Plot Loss/Epochs
            plu.generic_plot(f'PL Loss per nodi interni={train_info.hidden_layer_size} e alpha={train_info.alpha}', "Epochs", "Loss",
                             f'{plots}/pl_loss_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                                [range(1,train_info.max_epoch+2), train_info.train_loss, "Training Loss"],
                                         [range(1,train_info.max_epoch+2), train_info.val_loss, "Validation Loss"])
            # Plot Accuracy/Epochs
            plu.generic_plot(f'PL Accuracy per nodi interni={train_info.hidden_layer_size} e alpha={train_info.alpha}', "Epochs", "Accuracy",
                             f'{plots}/pl_accuracy_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                                [range(1, train_info.max_epoch + 2), train_info.train_accuracy, "Training Accuracy"],
                                         [range(1, train_info.max_epoch + 2), train_info.val_accuracy, "Validation Accuracy"])
        # NoES Plotting
        for train_info in std_results:
            # Plot Loss/Epochs
            plu.generic_plot(f'NoES Loss per nodi interni={train_info.hidden_layer_size}', "Epochs", "Loss",
                             f'{plots}/std_loss_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                                [range(1,train_info.max_epoch+2), train_info.train_loss, "Training Loss"],
                                         [range(1,train_info.max_epoch+2), train_info.val_loss, "Validation Loss"])
            # Plot Accuracy/Epochs
            plu.generic_plot(f'NoES Accuracy per nodi interni={train_info.hidden_layer_size}', "Epochs", "Accuracy",
                             f'{plots}/std_accuracy_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                                [range(1, train_info.max_epoch + 1), train_info.train_accuracy, "Training Accuracy"],
                                         [range(1, train_info.max_epoch + 1), train_info.val_accuracy, "Validation Accuracy"])

    print(f"[DEBUG] Training complete")


