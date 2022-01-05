# -*- coding: utf-8 -*-
import click
import logging
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

class Corrupt_MNIST(Dataset):
    def __init__(self,filepath,train=True):
        if train:
            print('Creating training set')
            self.train_path = filepath + '/train'
            file_list = glob.glob(self.train_path + "*")
            print('These files will be merged: \n',file_list)
            data = []
            targets = []
            for file in file_list:
                data_load = np.load(file)
                data.append(data_load['images'])
                targets.append(data_load['labels'])
            print('This is the len of data: ',len(data)) 
            self.images = torch.tensor(np.concatenate(data),dtype=torch.float)
            self.labels = torch.tensor(np.concatenate(targets),dtype=torch.float)
            print('Succesfully created trainset')
        else:
            print('Creating test set')
            self.test_path = filepath + '/test.npz'
            data = np.load(self.test_path)        
            self.images = torch.tensor(data['images'],dtype=torch.float)
            self.labels = torch.tensor(data['labels'],dtype=torch.float)
            print('Succesfully created testset')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        label = self.labels[idx]
        img = self.images[idx]
        return img,label

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def mnist(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    train = Corrupt_MNIST(input_filepath)
    torch.save(train,output_filepath + '/train.pth')
    test = Corrupt_MNIST(input_filepath,train=False)
    torch.save(test, output_filepath + '/test.pth')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    mnist()
