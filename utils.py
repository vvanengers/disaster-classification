import os
import time

import torch
from pathlib import Path


def save(obj, path, name, overwrite=True):
    Path(path).mkdir(parents=True, exist_ok=True)
    if not((not overwrite) & os.path.isfile(path+name)):
        torch.save(obj, path + name)
    else:
        # add time stamp to name if file is already there, and it shouldn't be overwritten
        torch.save(obj, path + name + str(time.time()))
