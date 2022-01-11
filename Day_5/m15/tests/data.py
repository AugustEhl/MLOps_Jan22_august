import torch
import numpy as np
from torch.utils.data import Dataset
import glob

class Corrupt_MNIST(Dataset):

    def __init__(self,filepath,train=True):
        if train:
            self.img_path = filepath
            file_list = glob.glob(self.img_path + 'train*')
            data = []
            targets = []
            for file in file_list:
                mnist_data = np.load(file)
                data.append(mnist_data['images'])
                targets.append(mnist_data['labels'])
            self.images = torch.tensor(np.concatenate(data),dtype=torch.float)
            self.labels = torch.tensor(np.concatenate(targets),dtype=torch.long)
        else:
            self.img_path = filepath
            file_list = glob.glob(self.img_path + 'test*')
            data = []
            targets = []
            for file in file_list:
                mnist_data = np.load(file)
                data.append(mnist_data['images'])
                targets.append(mnist_data['labels'])
            self.images = torch.tensor(np.concatenate(data),dtype=torch.float)
            self.labels = torch.tensor(np.concatenate(targets),dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        label = self.labels[idx]
        img = self.images[idx]
        return img,label

def mnist(filepath):
    # exchange with the corrupted mnist dataset
    train = Corrupt_MNIST(filepath)
    test = Corrupt_MNIST(filepath, train=False)
    return train, test

#torch.save(mnist('../../../dtu_mlops/data/corruptmnist/'),'data.pth')