import argparse
import sys
import numpy as np
import pickle

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from model import MyAwesomeModel
from torch.utils.data import Dataset

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

def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr',type=float, default=0.001)
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--weight_decay',type=float,default=1e-3)
    parser.add_argument('--data',type=str,default='data/processed/train.pth')
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=(0.85,0.89),weight_decay=args.weight_decay)
    train_set = torch.load(args.data)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    num_epochs = args.num_epochs
    train_loss = []
    train_acc = []
    for i in range(num_epochs):
        print('Epoch ',i+1,' of ',num_epochs)
        running_loss = 0
        running_acc = 0
        for images, labels in trainloader:
            labels = labels.long()
            #print(labels.shape)
            optimizer.zero_grad()   
            output = model(images)
            #print(output.shape)
            loss = criterion(output,labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            y_pred = (nn.Softmax(dim=1)(output)).argmax(dim=1)
            running_acc += torch.sum(y_pred == labels).item()/labels.shape[0]
        train_loss.append(running_loss)
        train_acc.append(running_acc/len(trainloader) * 100)
        print('Train loss: ', running_loss,
        '\ttrain accuracy: ', running_acc/len(trainloader) * 100,'%')
    torch.save(model, 'models/final_model.pth')
    pickle.dump(train_loss,open( "src/visualization/train_loss.p", "wb" ))
    pickle.dump(train_acc, open('src/visualization/train_acc.p', "wb"))

if __name__ == '__main__':
    train()