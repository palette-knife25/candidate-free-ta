from typing import Optional
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

class ExampleDataModule(pl.LightningDataModule):

    def __init__(self, data_root, batch_size: int = 32):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Normalize((0.1307,), (0.3081,))])
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.test_set = MNIST(self.data_root, train=False, download=True, transform=self.transform)
        full_set = MNIST(self.data_root, train=True, download=True, transform=self.transform)
        self.train_set, self.val_set = random_split(full_set, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
