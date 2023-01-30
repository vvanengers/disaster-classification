import numpy as np
from PIL import Image
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets


def load_data(root_dir='../data/Incidents-subset', test_size=0.2, batch_size=64, seed=42):
    # Define the transforms to apply to your images
    # In this example, we resize the images to 256x256 and normalize them
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Use the ImageFolder class to load the images from your root directory
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    mask = np.ones(len(dataset), dtype=bool)

    images_to_remove = []
    # Check if there are any corrupted images in the dataset
    for i, (image_path, c) in enumerate(dataset.imgs):
        try:
            # Open the image using the PIL library
            with Image.open(image_path) as image:
                image.verify()
        except (IOError, SyntaxError) as e:
            # If there is an error opening the image, it is likely corrupted
            # Remove it from the dataset
            print(f"Corrupted image: {image_path}")
            mask[i] = False
            images_to_remove.append((image_path, c))
    for x in images_to_remove:
        dataset.imgs.remove(x)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))

    train_indices, test_indices = indices[split:], indices[:split]

    # generate subset based on indices
    train_split = Subset(dataset, train_indices)
    test_split = Subset(dataset, test_indices)

    # create batches
    train_batches = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    test_batches = DataLoader(test_split, batch_size=batch_size)
    # return sample_dist for adjusting the loss value based on the image counts

    index_to_names = {i: name for i, name in enumerate(dataset.classes)}
    labels, sample_dist = np.unique(dataset.targets, return_counts=True)
    names = [index_to_names[label] for label in labels]

    return train_batches, test_batches, sample_dist, names
