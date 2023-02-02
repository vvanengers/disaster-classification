import copy

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, datasets
from utils import print_and_log


def load_data(root_dir='./data/Incidents-subset', val_size=0.05, test_size=0.05, batch_size=64, seed=42):
    dataset = load_data_from_folder(root_dir)

    # Shuffle dataset
    s_ind = (list(range(len(dataset))))
    np.random.shuffle(s_ind)
    dataset.imgs = np.array(dataset.imgs)[s_ind]
    dataset.targets = np.array(dataset.targets)[s_ind]

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split1 = int(dataset_size * (1 - (val_size + test_size)))
    split2 = int(dataset_size * test_size) + split1


    train_indices = indices[:split1]
    val_indices = indices[split1:split2]
    test_indices = indices[split2:]

    train_data_set = copy.deepcopy(dataset)
    train_data_set.imgs = train_data_set.imgs[:split1]
    train_data_set.targets = train_data_set.targets[:split1]

    # generate subset based on indices
    val_split = Subset(dataset, val_indices)
    test_split = Subset(dataset, test_indices)

    # rest for testing
    labels = np.unique(dataset.targets)
    _, sample_dist = np.unique(train_data_set.targets, return_counts=True)
    weight = 1/(sample_dist / np.sum(sample_dist))
    samples_weight = weight[train_data_set.targets]

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(train_data_set))

    # create batches
    train_batches = DataLoader(train_data_set, batch_size=batch_size, sampler=sampler)
    val_batches = DataLoader(val_split, batch_size=batch_size)
    test_batches = DataLoader(test_split, batch_size=batch_size)
    # return sample_dist for adjusting the loss value based on the image counts

    index_to_names = {i: name for i, name in enumerate(dataset.classes)}
    names = [index_to_names[label] for label in labels]

    return train_batches, val_batches, test_batches, names, weight


def load_data_from_folder(root_dir):
    # print_and_log('Start data loading')
    # Define the transforms to apply to your images
    # In this example, we resize the images to 256x256 and normalize them
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Use the ImageFolder class to load the images from your root directory
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    remove_corrupted_images(dataset)
    return dataset


def remove_corrupted_images(dataset):
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
        dataset.targets.remove(x[1])

