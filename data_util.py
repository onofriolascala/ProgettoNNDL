import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_train_and_test_set_loader(data_path, trainset_len, testset_len):

    transform = transforms.ToTensor()  # Convert images to tensors
    # Load the full dataset
    full_mnist = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)

    generator = torch.Generator().manual_seed(42)

    train_subset, test_subset = random_split(full_mnist, [trainset_len, testset_len], generator=generator)

    print("[DEBUG] Train_subset len: ", len(train_subset))
    print("[DEBUG] Test_subset len: ", len(test_subset))
    # Print basic information about the dataset
    print(f"[DEBUG] Full dataset size: {len(full_mnist)} samples")

    train_loader = DataLoader(train_subset, batch_size=trainset_len, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=testset_len, shuffle=False, num_workers=0)

    return train_loader, test_loader


