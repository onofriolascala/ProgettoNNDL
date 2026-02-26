import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten dei dati
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


def compute_k(epochs):
    root = int(math.sqrt(epochs))
    for k in range(root, 0, -1):
        if epochs % k == 0:
            return k
    return root



def train_net(train_loader, test_loader, net, device, optimizer, alpha, GL=True, loss_fn=nn.CrossEntropyLoss(), epochs=1000):

    train_loss = []
    val_loss = []

    best_val_loss = np.inf
    best_train_loss = np.inf
    strip_train_loss = []
    k = compute_k(epochs)
    stop_criteria_value = 0

    for epoch in range(epochs):
        net.train()
        epoch_train_loss = 0.0
        for (data, labels) in train_loader:
            data, labels = data.to(device), labels.to(device)

            output = net(data)

            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(epoch_train_loss)

        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss

        if not GL:
            strip_train_loss.append(epoch_train_loss)


        net.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():

            for (data, labels) in test_loader:

                data, labels = data.to(device), labels.to(device)

                output = net(data)

                loss = loss_fn(output, labels)

                epoch_val_loss += loss.item()


            epoch_val_loss = epoch_val_loss / len(test_loader)

            val_loss.append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

            gl_value = 100 * ((epoch_val_loss / best_val_loss) - 1)

            if GL:
                stop_criteria_value = gl_value
            else:
                if len(strip_train_loss) >= k:
                    pk = 1000 * ((sum(strip_train_loss) / (k * best_train_loss)) - 1)
                    stop_criteria_value = gl_value / pk
                    #reset the strip
                    strip_train_loss = []

            if stop_criteria_value > alpha:
                return train_loss, val_loss

    return train_loss, val_loss







