import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_train_and_test_set_loader(data_path, trainset_len, testset_len, train_batch_size, test_batch_size):

    train_loader=get_train_loader(data_path, trainset_len, train_batch_size)
    test_loader=get_test_loader(data_path, testset_len, test_batch_size)

    return train_loader, test_loader

def get_train_loader(data_path, trainset_len, batch_size):

    transform = transforms.ToTensor()
    full_mnist = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    generator = torch.Generator().manual_seed(42)
    train_subset, _ = random_split(full_mnist, [trainset_len, len(full_mnist)-trainset_len], generator=generator)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader

def get_test_loader(data_path, testset_len, batch_size):

    transform = transforms.ToTensor()
    full_mnist = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    generator = torch.Generator().manual_seed(42)
    test_subset, _ = random_split(full_mnist, [testset_len, len(full_mnist) - testset_len], generator=generator)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=0)

    return test_loader

