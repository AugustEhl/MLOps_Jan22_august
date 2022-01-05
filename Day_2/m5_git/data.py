import torch
import numpy as np
from torch.utils.data import Dataset

class Corrupt_MNIST(Dataset):

    def __init__(self,train=True):
        if train:
            data0 = np.load('../../../data/corruptmnist/train_0.npz')
            data1 = np.load('../../../data/corruptmnist/train_1.npz')
            data2 = np.load('../../../data/corruptmnist/train_2.npz')
            data3 = np.load('../../../data/corruptmnist/train_3.npz')
            data4 = np.load('../../../data/corruptmnist/train_4.npz')
            self.images = torch.tensor(np.concatenate((data0['images'],data1['images'],data2['images'],data3['images'],data4['images'])),dtype=torch.float)
            self.labels = torch.tensor(np.concatenate((data0['labels'],data1['labels'],data2['labels'],data3['labels'],data4['labels'])),dtype=torch.float)
        else:
            test = np.load('../../../data/corruptmnist/test.npz')
            self.images = torch.tensor(test['images'],dtype=torch.float)
            self.labels = torch.tensor(test['labels'],dtype=torch.float)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        label = self.labels[idx]
        img = self.images[idx]
        return img,label

def mnist():
    # exchange with the corrupted mnist dataset
    train = Corrupt_MNIST()
    test = Corrupt_MNIST(train=False)
    return train, test