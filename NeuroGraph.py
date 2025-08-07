import torch
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip
)
class NeuroGraphDynamic():
    r"""Graph-based neuroimaging benchmark datasets, e.g.,
        :obj:`"DynHCPGender"`, :obj:`"DynHCPAge"`, :obj:`"DynHCPActivity"`,
        :obj:`"DynHCPWM"`, or :obj:`"DynHCPFI"`

        Args:
            root (str): Root directory where the dataset should be saved.
            name (str): The name of the dataset.

        Returns:
            list: A list of graphs in PyTorch Geometric (pyg) format. Each graph contains a list of dynamic graphs batched in pyg batch.
    """
    url = 'https://vanderbilt.box.com/shared/static'
    filenames = {
        'DynHCPGender': 'mj0z6unea34lfz1hkdwsinj7g22yohxn.zip',
        'DynHCPActivity': '2so3fnfqakeu6hktz322o3nm2c8ocus7.zip',
        'DynHCPAge': '195f9teg4t4apn6kl6hbc4ib4g9addtq.zip',
        'DynHCPWM': 'mxy8fq3ghm60q6h7uhnu80pgvfxs6xo2.zip',
        'DynHCPFI': 'un7w3ohb2mmyjqt1ou2wm3g87y1lfuuo.zip',
    }

    def __init__(self, root, name):
        self.root = root
        self.name = name
        assert name in self.filenames.keys()
        self.name = name
        file_path = os.path.join(self.root, self.name, 'processed', self.name + ".pt")
        if not os.path.exists(file_path):
            self.download()
        self.dataset, self.labels = self.load_data()





    def download(self):
        url = f'{self.url}/{self.filenames[self.name]}'
        path = download_url(url, os.path.join(self.root, self.name))
        extract_zip(path, self.root)
        os.unlink(path)



    def load_data(self):
        if self.name == 'DynHCPActivity':
            #dataset_raw = torch.load(os.path.join(self.root, self.name, 'processed', self.name + ".pt"))
            dataset_raw = torch.load(
                os.path.join(self.root, self.name, 'processed', self.name + ".pt"),
                weights_only=False
            )

            dataset, labels = [], []
            for v in dataset_raw:
                batches = v.get('batches')
                if len(batches) > 0:
                    for b in batches:
                        y = b.y[0].item()
                        dataset.append(b)
                        labels.append(y)
        else:
            dataset = torch.load(os.path.join(self.root, self.name, 'processed', self.name + ".pt"),weights_only=False)
            labels = dataset['labels']
            dataset = dataset['batches']
        return dataset, labels