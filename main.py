
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split

input_size = 28*28
output_size = 10

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      # First fully connected layer
      self.fc1 = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = '/tmp/data'  # Root directory where data will be stored
    transform = transforms.ToTensor()  # Convert images to tensors
    # Load the full dataset
    full_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    full_trainset = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    generator = torch.Generator().manual_seed(42)

    train_subset, _ = random_split(full_dataset, [10000, len(full_dataset)-10000], generator=generator)
    test_subset, _ = random_split(full_trainset, [2500, len(full_trainset)-2500], generator=generator)

    print("[DEBUG] train_subset len: ", len(train_subset))
    print("[DEBUG] test_subset len: ", len(test_subset))

    # criterion = nn.
    # optimizer = torch.optim.Rprop()

    nn = Net()

    for epoch in range(100):
        print(f'\nEpoch {epoch + 1}')




    # Print basic information about the dataset
    print(f"Full dataset size: {len(full_dataset)} samples")