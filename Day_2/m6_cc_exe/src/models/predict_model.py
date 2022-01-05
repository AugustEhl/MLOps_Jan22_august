import numpy as np
import pickle
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
import click
import logging
import glob
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
@click.argument('pretrained_model', type=click.Path(exists=True))
@click.argument('test_data', type=click.Path())
def evaluate(pretrained_model,test_data):
    """
    Tests model on test data in data/processed
    """
    logger = logging.getLogger(__name__)
    logger.info('Evaluating pretrained model on test data')
    print("Evaluating until hitting the ceiling")
    
    # TODO: Implement evaluation logic here
    model = torch.load(pretrained_model)
    test_set = torch.load(test_data)
    images = test_set.images
    labels = test_set.labels
    labels = labels.long()
    model.eval()
    with torch.no_grad():
        output = model(images)
        y_pred = (nn.Softmax(dim=1)(output)).argmax(dim=1)
    accuracy = torch.sum(y_pred == labels).item()/len(labels)*100
    print('The test accuracy is: ', accuracy,'%')
    pickle.dump([accuracy], open('src/visualization/test_acc.p', "wb"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    evaluate()