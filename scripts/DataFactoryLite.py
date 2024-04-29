import torch
from torch.utils.data import Dataset
import json
import numpy as np
import pytorch_lightning as pl

class RA_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        truths = [torch.nan]*11
        truths[-1] = item[0][-1]
        return {'x': torch.tensor(np.array(item[1])), 'y': torch.tensor(np.array(truths))}

class WrapperDataModule(pl.LightningDataModule):
    def __init__(self, dl_train, dl_val, dl_test):
        super().__init__()
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.dl_test = dl_test

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_val

    def test_dataloader(self):
        return self.dl_test

def trivial_collate_fn(batch):
    return batch

def random_anomaly(home_dir):
    data_dir = home_dir+'data/'

    train_dataset = json.load(open(data_dir+'RandomAnomaly_Train.json'))
    validate_dataset = json.load(open(data_dir+'RandomAnomaly_Validate.json'))
    test_dataset = json.load(open(data_dir+'RandomAnomaly_Test.json'))

    RA_train = RA_Dataset(train_dataset)
    RA_validate = RA_Dataset(validate_dataset[:1000])
    RA_test = RA_Dataset(test_dataset[:1000])

    dl_train = torch.utils.data.DataLoader(RA_train, num_workers=0, batch_size=16, collate_fn = trivial_collate_fn)
    dl_val = torch.utils.data.DataLoader(RA_validate, num_workers=0, batch_size=16, collate_fn = trivial_collate_fn)
    dl_test = torch.utils.data.DataLoader(RA_test, num_workers=0, batch_size=16, collate_fn = trivial_collate_fn)
    
    return WrapperDataModule(dl_train, dl_val, dl_test)










