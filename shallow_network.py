import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        # First fully connected layer
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten dei dati
        x = self.fc1(x)
        x = f.sigmoid(x)
        x = self.fc2(x)
        return x


def compute_k(epochs):
    root = int(math.sqrt(epochs))
    for k in range(root, 0, -1):
        if epochs % k == 0:
            return k
    return root



def train_net(train_loader, test_loader, net, device, optimizer, loss_fn=nn.CrossEntropyLoss(), early_stopping=True, alpha = 1, use_gl=True, epochs=5000):

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    best_val_loss = np.inf
    best_train_loss = np.inf
    best_train_loss_epoch = 0
    best_val_loss_epoch = 0
    strip_train_loss = []
    k = compute_k(epochs)
    stop_criteria_value = 0

    print(f'[DEBUG] Starting training with number of nodes: {net.hidden_size} GL: {use_gl} alpha: {alpha} early_stopping: {early_stopping}')

    for epoch in range(epochs):
        net.train()
        epoch_train_loss = 0.0
        correct = 0
        total = 0
        if epoch % 1000 == 0:
            print("Starting epoch ", epoch)
        for (data, labels) in train_loader:
            data, labels = data.to(device), labels.to(device)

            output = net(data)

            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            predictions = torch.argmax(output, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy.append(correct/total)

        epoch_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(epoch_train_loss)

        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
            best_train_loss_epoch = epoch

        if not use_gl:
            strip_train_loss.append(epoch_train_loss)


        net.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():

            for (data, labels) in test_loader:

                data, labels = data.to(device), labels.to(device)

                output = net(data)

                loss = loss_fn(output, labels)

                epoch_val_loss += loss.item()

                predictions = torch.argmax(output, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            val_accuracy.append(correct/total)

            epoch_val_loss = epoch_val_loss / len(test_loader)

            val_loss.append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_loss_epoch = epoch

            gl_value = 100 * ((epoch_val_loss / best_val_loss) - 1)

            if early_stopping:
                if use_gl:
                    stop_criteria_value = gl_value
                else:
                    if len(strip_train_loss) >= k:
                        pk = 1000 * ((sum(strip_train_loss) / (k * best_train_loss)) - 1)
                        stop_criteria_value = gl_value / pk
                        #reset the strip
                        strip_train_loss = []

                if stop_criteria_value > alpha:
                    #log training stopped at epoch: %epoch
                    return train_loss, val_loss, train_accuracy, val_accuracy, best_train_loss_epoch, best_val_loss_epoch, epoch
            else:
                strip_train_loss = []

    return train_loss, val_loss, train_accuracy, val_accuracy, best_train_loss_epoch, best_val_loss_epoch, epochs

@dataclass
class TrainInfo:
    train_loss: list
    val_loss: list
    train_accuracy: list
    val_accuracy: list
    max_epoch: int
    hidden_layer_size: int
    alpha: float
    best_train_loss_epoch: int
    best_val_loss_epoch: int

def get_training_info(train_loader, test_loader, net, device, optimizer, loss_fn=nn.CrossEntropyLoss(), early_stopping=True, alpha = 1, use_gl=True, epochs=5000):
    train_loss, val_loss, train_accuracy, val_accuracy, best_train_loss_epoch, best_val_loss_epoch, max_epoch = train_net(train_loader, test_loader, net, device, optimizer, loss_fn, early_stopping, alpha, use_gl, epochs)
    train_info = TrainInfo(
        train_loss,
        val_loss,
        train_accuracy,
        val_accuracy,
        max_epoch,
        net.hidden_size,
        alpha,
        best_train_loss_epoch,
        best_val_loss_epoch,
    )
    return train_info





