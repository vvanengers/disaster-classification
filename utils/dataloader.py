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


def load_folded_dataloaders(dataset, k_folds=5, batch_size=64, seed=42):
    l = len(dataset)
    fold_size = int(l / k_folds)
    last_fold = l - (k_folds - 1) * fold_size
    assert l == fold_size * (k_folds - 1) + last_fold
    dataset_folds = random_split(dataset, [fold_size] * (k_folds - 1) + [last_fold],
                                 generator=torch.Generator().manual_seed(42))
    dataloaders = []
    for k in range(k_folds):
        print('Loading fold ',k)
        # get all folds except the kth
        train_folds = dataset_folds[:k] + dataset_folds[k+1:]
        valid_fold = dataset_folds[k]
        train_folds = torch.utils.data.ConcatDataset(train_folds)

        # targets = [c for _, c, __ in train_folds]
        targets = np.array(dataset.targets)[[i for ds in train_folds.datasets for i in ds.indices]]

        # rest for testing
        labels, class_counts = np.unique(targets, return_counts=True)
        weight = 1 / torch.tensor(class_counts).float()
        samples_weight = weight[targets]

        sampler = WeightedRandomSampler(samples_weight, len(train_folds))

        # create batches
        train_loader = DataLoader(train_folds, batch_size=batch_size, sampler=sampler)
        valid_loader = DataLoader(valid_fold, batch_size=batch_size)
        dataloaders.append((train_loader, valid_loader))
    return dataloaders


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
