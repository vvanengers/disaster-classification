import copy
import logging
import os
import time

import torch
from pathlib import Path

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
