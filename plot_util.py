import matplotlib.pyplot as plt
import numpy as np

def generic_plot(title, xlabel, ylabel, path, *data_sets):
    """
    Saves to system a jpg figure of the desired plot graph. The function takes an indefinite number of data
    sets and, if properly codified, draws the corresponding number of plots.
    :param title: title of figure
    :param xlabel: xlabel of figure
    :param ylabel: ylabel of figure
    :param data_sets: each data set is a tuple (x,y,label) where x are the x coordinates, y are the y coordinates and label is the data label for the plot
    """
    plt.figure()
    legend = []

    for data_set in data_sets:
        plt.plot(data_set[0], data_set[1], label = data_set[2])
        legend.append(data_set[2])

    #plt.axis((0, 125, 0, 330))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(legend)
    plt.grid(linewidth=0.3)
    #plt.show()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def generic_plotv2(title, xlabel, ylabel, path, *data_sets):
    """
    Saves to system a jpg figure of the desired plot graph. The function takes an indefinite number of data
    sets and, if properly codified, draws the corresponding number of plots.
    :param title: title of figure
    :param xlabel: xlabel of figure
    :param ylabel: ylabel of figure
    :param data_sets: each data set is a tuple (x,y,label) where x are the x coordinates, y are the y coordinates and label is the data label for the plot
    """
    plt.figure()
    legend = []

    for data_set in data_sets:
        plt.plot(data_set[0], data_set[1], label = data_set[2])
        if not data_set[2]:
            legend.append(data_set[2])

    #plt.axis((0, 125, 0, 330))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if len(legend)>0:
        plt.legend(legend)

    plt.grid(linewidth=0.3)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def trace_data(results, stopping_type, path):
    for train_info in results:
        # Plot Loss/Epochs
        generic_plot(f'Loss {stopping_type} per nodi interni={train_info.hidden_layer_size} e alpha={train_info.alpha}',
                         "Epochs", "Loss",
                         f'{path}/{stopping_type}_loss_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                         [range(1, train_info.max_epoch + 2), train_info.train_loss, "Training Loss"],
                         [range(1, train_info.max_epoch + 2), train_info.val_loss, "Validation Loss"])
        # Plot Accuracy/Epochs
        generic_plot(f'Accuracy {stopping_type} per nodi interni={train_info.hidden_layer_size} e alpha={train_info.alpha}',
                         "Epochs", "Accuracy",
                         f'{path}/gl_accuracy_{train_info.hidden_layer_size}-{train_info.alpha}.jpg',
                         [range(1, train_info.max_epoch + 2), train_info.train_accuracy, "Training Accuracy"],
                         [range(1, train_info.max_epoch + 2), train_info.val_accuracy, "Validation Accuracy"])

def test_plot():
    x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
    y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

    temp_data = [x,y,"Temp"]
    generic_plot("Title", "Xlabel", "Ylabel", temp_data)