import sys
import numpy as np
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
sys.path.insert(0,'src/models')

import torch
from torch import nn
from torch.utils.data import Dataset

import click
import logging
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
@click.argument('pretrained_model', type=click.Path())
@click.argument('train_loss', type=click.Path())
@click.argument('train_acc', type=click.Path())
@click.argument('test_acc', type=click.Path())
@click.argument('train_data', type=click.Path())
def visualize(pretrained_model,train_loss,train_acc,test_acc,train_data):
    """
    visualize results
    """
    logger = logging.getLogger(__name__)
    logger.info('Visualizing training and validation')
    print(pretrained_model)
    
    # TODO: Implement evaluation logic here
    model = torch.load(pretrained_model)
    model.eval()
    model_pars = list(model.parameters())[-1].data.numpy()
    trainloss = pickle.load(open(train_loss,"rb"))
    trainacc = pickle.load(open(train_acc,"rb"))    
    accuracy = pickle.load(open(test_acc,"rb"))[0]
    print('The test accuracy is: ', accuracy,'%')
    plt.plot(range(1,len(trainloss)+1),trainloss,'-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train loss')
    plt.savefig('reports/figures/Train_loss.png')
    plt.close()
    plt.plot(range(1,len(trainacc)+1),trainacc,'-')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Train Accuracy')
    plt.savefig('reports/figures/Train_acc.png')
    plt.close()
    plt.bar(x = range(1,len(model_pars)+1),height=model_pars)
    plt.xlabel('weights')
    plt.ylabel('Size of weight')
    plt.title('Weights in final layer')
    plt.savefig('reports/figures/weights_final_layer.png')
    plt.close()
    traindata = torch.load(train_data).images
    traindata = traindata.view(traindata.shape[0],-1).data.numpy()
    print(traindata.shape)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(traindata)
    print(X_embedded)
    plt.plot(X_embedded[:,0],X_embedded[:,1],'*')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('t-SNE')
    plt.savefig('reports/figures/t_SNE.png')
    plt.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    visualize()