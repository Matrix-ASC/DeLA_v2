import os
import h5py
import random, math
import torch
from torch.utils.data import Dataset
from config import data_path
from var import get_mix_prob

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class ScanObjectNN(Dataset):
    def __init__(self, partition='training'):
        h5_name = data_path / f"main_split/{partition}_objectdataset_augmentedrot_scale75.h5"
        f = h5py.File(h5_name, mode="r")
        self.data = torch.from_numpy(f['data'][:]).float()
        self.label = torch.from_numpy(f['label'][:]).type(torch.uint8)
        f.close()
        self.partition = partition

    def get(self, idx):
        pc = self.data[idx]
        label = self.label[idx]
        if self.partition == 'training':
            angle = random.random() * 2 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotmat = torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
            scale = torch.rand((3,)) * 0.2 + 0.9
            rotmat = torch.diag(scale) @ rotmat
            pc = pc @ rotmat
            pc = pc[torch.randperm(pc.shape[0])]

        return pc.mul(40), label

    def __getitem__(self, idx):
        if self.partition == 'training':
            pca, lba = self.get(idx)
            pcb, lbb = self.get(random.randrange(0, len(self)))
            if random.random() < get_mix_prob():
                crop = random.randint(0, 1024)
            else:
                crop = 0
            pc = torch.cat([pca[crop:], pcb[:crop]], dim=0)
            # noise suppression
            ca = (2048 - crop) / 128 - 2
            ca = (1 - crop / 2048) / (1 + math.exp(-ca))
            cb = crop / 128 - 2
            cb = (crop / 2048) / (1 + math.exp(-cb))
            cs = ca + cb
            label = torch.zeros(15) + 0.2/15
            label[lba.item()] += 0.8 * ca / cs
            label[lbb.item()] += 0.8 * cb / cs
            return pc, label
        else:
            return self.get(idx)

    def __len__(self):
        return self.data.shape[0]

