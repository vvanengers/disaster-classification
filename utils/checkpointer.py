import torch

import utils


class Checkpointer:
    def __init__(self, store_path, store_name, args, overwrite=True, autosave=True) -> None:
        super().__init__()
        self.store_path = store_path
        self.store_name = store_name
        self.overwrite = overwrite
        self.autosave = autosave
        self.store = {'args': args}

    def add_in_list(self, key, obj):
        if key not in self.store:
            self.store[key] = []
        self.store[key].append(obj)
        if self.autosave:
            self.save()

    def add_singular(self, key, obj):
        self.store[key] = obj
        if self.autosave:
            self.save()

    def save(self):
        utils.save(self.store, self.store_path, self.store_name, overwrite=self.overwrite)

    def load(self, path):
        self.store = torch.load(path)

