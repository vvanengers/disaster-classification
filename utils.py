import copy
import logging
import os
import random
import time

import numpy as np
import skimage
import torch
from pathlib import Path

from skimage import io
from skimage.filters import gaussian
from skimage.transform import rotate, warp
from skimage.util import random_noise

logger = None


def save(obj, path, name, overwrite=True):
    Path(path).mkdir(parents=True, exist_ok=True)
    if not((not overwrite) & os.path.isfile(path+name)):
        torch.save(obj, path + name)
    else:
        # add time stamp to name if file is already there, and it shouldn't be overwritten
        torch.save(obj, path + name + str(time.time()))


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    Path('logs/').mkdir(parents=True, exist_ok=True)
    log_path = f'logs/{time.strftime("%Y%m%d%H%M%S")}'

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


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

    setup_logger('')
    print_and_log('Start augmenting dataset')
    dataset_size = len(dataset)
    for index in range(dataset_size):
        if index % 500 == 0:
            print_and_log(f'{index}/{dataset_size}')
        path = dataset.imgs[index][0]
        image = io.imread(path)
        io.imshow(image)
        for name, transformation in transformations.items():
            new_path = add_string_before_filetype(path, '_' + name)
            io.imsave(add_string_before_filetype(new_path, name), transformation(image))
