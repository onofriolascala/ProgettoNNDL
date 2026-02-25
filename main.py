
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

input_size = 28*28
hidden_layer_size = 32
output_size = 10

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.fc1(x) # equivale ad a
        x = F.sigmoid(x) # g(a) = y
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    root = './data'  # Root directory where data will be stored
    transform = transforms.ToTensor()  # Convert images to tensors
    # Load the full dataset
    full_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    full_testset = datasets.MNIST(root=root, train=False, transform=transform, download=True)

    generator = torch.Generator().manual_seed(42)

    train_subset, _ = random_split(full_dataset, [10000, len(full_dataset)-10000], generator=generator)
    test_subset, _ = random_split(full_testset, [2500, len(full_testset) - 2500], generator=generator)

    print("[DEBUG] Train_subset len: ", len(train_subset))
    print("[DEBUG] Test_subset len: ", len(test_subset))
    # Print basic information about the dataset
    print(f"Full dataset size: {len(full_dataset)} samples")

    my_network = Net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Rprop(my_network.parameters(), lr=0.01)

    shuffle = True
    batch_size = len(train_subset)

    trainset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)

    batch_size = len(test_subset)
    validation_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=shuffle)

    print(f"[DEBUG] Subset DataLoader: {len(trainset_loader)} batches per epoch")

    my_network.train()

    for epoch in range(100):

        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(trainset_loader):
            # Flatten the images
            data = data.view(-1, input_size)
            #Transform the label in one hot coding
            #labels_onehot = F.one_hot(labels, num_classes=10).float()
            # Forward pass
            outputs = my_network(data)
            # loss = loss_criterion(outputs, labels_onehot)
            loss = loss_criterion(outputs, labels)


            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss: {loss.item():.4f}')

        my_network.eval()

        valid_loss = 0.0
        for data, labels in validation_loader:
            data = data.view(-1, input_size)
            labels_onehot = F.one_hot(labels, num_classes=10).float()
            # Forward Pass
            target = my_network(data)
            # Find the Loss
            loss = loss_criterion(target, labels_onehot)
            # Calculate Loss
            valid_loss += loss.item()

        print(f'Epoch {epoch + 1} \t\t Training Loss: { \
            train_loss / len(trainset_loader)} \t\t Validation Loss: { \
            valid_loss / len(validation_loader)}')

    print(f"[DEBUG] Training complete")





