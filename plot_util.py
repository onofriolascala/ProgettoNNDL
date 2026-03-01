import matplotlib.pyplot as plt

def generic_plot(title, xlabel, ylabel, best_epoch, path, *data_sets):
    """
    Saves to system a jpg figure of the desired plot graph. The function takes an indefinite number of data
    sets and, if properly codified, draws the corresponding number of plots.
    :param title: title of figure
    :param xlabel: xlabel of figure
    :param ylabel: ylabel of figure
    :param best_epoch: x-axis integer for best training epoch
    :param path: file path where to save the figure
    :param data_sets: each data set is a tuple (x,y,label) where x are the x coordinates, y are the y coordinates and label is the data label for the plot
    """
    plt.figure()
    legend = []

    for data_set in data_sets:
        plt.plot(data_set[0], data_set[1], label = data_set[2])
        if data_set[2] != "":
            legend.append(data_set[2])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if best_epoch != 0:
        plt.axvline(best_epoch, linewidth = 0.5, label = 'Best Epoch')
        legend.append('Best Epoch')
    if len(legend)>0:
        plt.legend(legend)

    plt.grid(linewidth = 0.3)
    plt.savefig(path, bbox_inches = 'tight')
    plt.close()

def trace_data(results, stopping_type, path):
    for train_info in results:
        if stopping_type.find("NoES"):
            title = f'{stopping_type} con nodi interni={train_info.hidden_layer_size} e alpha={train_info.alpha}'
        else:
            title = f'{stopping_type} con nodi interni={train_info.hidden_layer_size}'
        # Plot Loss/Epochs
        generic_plot(f'Loss {title}',
                         "Epochs", "Loss", train_info.best_val_loss_epoch,
                         f'{path}/{stopping_type}/{stopping_type}_loss_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                         [range(1, len(train_info.train_loss)+1), train_info.train_loss, "Training Loss"],
                                  [range(1, len(train_info.val_loss) + 1), train_info.val_loss, "Validation Loss"])
        # Plot Accuracy/Epochs
        generic_plot(f'Accuracy {title}',
                         "Epochs", "Accuracy", train_info.best_val_loss_epoch,
                         f'{path}/{stopping_type}/{stopping_type}_accuracy_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                         [range(1, len(train_info.train_accuracy)+1), train_info.train_accuracy, "Training Accuracy"], #train_info.max_epoch + 1
                                  [range(1, len(train_info.val_accuracy) + 1), train_info.val_accuracy, "Validation Accuracy"])