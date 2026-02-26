import matplotlib.pyplot as plt
import numpy as np

def generic_plot(title, xlabel, ylabel, *data_sets):
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
    plt.show()
    #plt.savefig('plots/foo.jpg', bbox_inches='tight')
    plt.close()

def test_plot():
    x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
    y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

    temp_data = [x,y,"Temp"]
    generic_plot("Title", "Xlabel", "Ylabel", temp_data)