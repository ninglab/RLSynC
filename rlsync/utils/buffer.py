import random
import torch
import os

class Buffer(object):
    def __init__(self, size):
        self._list = []
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index < len(self._list):
            return self._list[index]
        else:
            return None

    def append(self, item):
        self._list.insert(0, item)
        if len(self._list) > self.size:
            self._list.pop()
    
    def __add__(self, other):
        newlist = other._list + self._list
        newbuff = Buffer(self.size)
        newbuff._list = newlist[:self.size]
        return newbuff

    def __iadd__(self, other):
        newlist = other._list + self._list
        self._list = newlist[:self.size]
        return self

    def sample(self, k):
        return random.sample(self._list, k)

class PreloadedBuffer(Buffer):
    def __init__(self, size, ground_truth="data/rpb"):
        super().__init__(size)
        self._list = self.load(ground_truth)
        self._list = self._list[:self.size]
    
    def load(self, datafolder):
        data = []
        for fn in os.listdir(datafolder):
            if "rpb_" in fn and ".pt" in fn:
                data.append(torch.load(datafolder+"/"+fn)[-1])
        return data