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
        x = F.sigmoid(x) #Da modificare con identità?
        return x

def train_net(train_loader, test_loader, net, device, optimizer, alpha, GL=True, loss_fn=nn.CrossEntropyLoss(), epochs=1000):
    net.train()
    train_loss = []
    val_loss = []
    best_val_loss = np.inf
    best_train_loss = np.inf
    strip_train_loss = []

    for epoch in range(epochs):
        for (data, labels) in train_loader:
            data, labels = data.to(device), labels.to(device)

            output = net(data)

            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if loss.item() < best_train_loss:
                best_train_loss = loss.item()

            if not GL:
                strip_train_loss.append(loss.item())


        for (data, labels) in test_loader:

            data, labels = data.to(device), labels.to(device)

            output = net(data)

            loss = loss_fn(output, labels)

            val_loss.append(loss.item())

            if loss.item() < best_val_loss:
                best_val_loss = loss.item()

        if GL:





