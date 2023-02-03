import copy

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, datasets


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]
        return img, label, path


def load_data(root_dir='./data/Incidents-subset', val_size=0.05, test_size=0.05, batch_size=64, seed=42):
    dataset = load_data_from_folder(root_dir)


    dsz = len(dataset)

    num_train = int(dsz*(1-(val_size + test_size)))
    num_val = int(dsz*val_size)
    num_test = dsz - (num_train + num_val)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test],
                                                            generator=torch.Generator().manual_seed(42))

    targets = [c for _, c, __ in train_dataset]



    # rest for testing
    labels, class_counts = np.unique(targets, return_counts=True)
    weight = 1 / torch.tensor(class_counts).float()
    samples_weight = weight[targets]

    sampler = WeightedRandomSampler(samples_weight, len(train_dataset))

    # create batches
    train_batches = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_batches = DataLoader(val_dataset, batch_size=batch_size)
    test_batches = DataLoader(test_dataset, batch_size=batch_size)
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
    dataset = ImageFolderWithPaths(root_dir, transform=transform)
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

