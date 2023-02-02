import argparse
import random

import numpy as np
import skimage
from skimage import io
from skimage.filters import gaussian
from skimage.transform import rotate, warp
from skimage.util import random_noise

import dataloader
from utils import setup_logger, print_and_log


def image_augmentation(dataset):
    def find_last(lst, sought_elt):
        for r_idx, elt in enumerate(reversed(lst)):
            if elt == sought_elt:
                return len(lst) - 1 - r_idx

    def add_string_before_filetype(orig, addition):
        period = find_last(orig, '.')
        return orig[:period] + addition + orig[period:]

    matrix = np.array([[random.randint(0, 100) / 100, -0.1, 0],
                       [0, random.randint(8, 10) / 10, 0],
                       [0, random.randint(-100, 1000) / 1000000, 1]])
    tform = skimage.transform.ProjectiveTransform(matrix=matrix)
    transformations = {
        'rotate': lambda image: rotate(image, angle=random.randint(0, 360), mode='constant'),
        'wrapshift': lambda image: warp(image, tform.inverse),
        'fliplr': lambda image: np.fliplr(image),
        'flipud': lambda image: np.flipud(image),
        'noisyrandom': lambda image: random_noise(image, var=0.5 ** (random.randint(3, 6))),
        'gaussian': lambda image: gaussian(image, sigma=(random.randint(0, 5) * 2 + 1), multichannel=True)
    }

    # setup_logger('')
    print('Start augmenting dataset')
    dataset_size = len(dataset)
    for index in range(dataset_size):
        if index % 500 == 0:
            print(f'{index}/{dataset_size}')
        path = dataset.imgs[index][0]
        image = io.imread(path)
        io.imshow(image)
        for name, transformation in transformations.items():
            new_path = add_string_before_filetype(path, '_' + name)
            io.imsave(add_string_before_filetype(new_path, name), transformation(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image augmentation')
    parser.add_argument('--path', type=str, default=None, help='Image folder path')
    args = parser.parse_args()
    dataset = dataloader.load_data_from_folder(args.path)
    image_augmentation(dataset)
