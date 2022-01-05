import argparse
import sys
import numpy as np
import pickle

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms

from data import mnist,Corrupt_MNIST
from model import MyAwesomeModel

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr',type=float, default=0.001)
        parser.add_argument('--num_epochs',type=int,default=30)
        parser.add_argument('--batch_size',type=int,default=64)
        parser.add_argument('--weight_decay',type=float,default=1e-3)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=(0.85,0.89),weight_decay=args.weight_decay)
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        num_epochs = args.num_epochs
        for i in range(num_epochs):
            print('Epoch ',i+1,' of ',num_epochs)
            running_loss = 0
            running_acc = 0
            train_loss = []
            train_acc = []
            for images, labels in trainloader:
                labels = labels.long()
                #print(labels.shape)
                optimizer.zero_grad()   
                output = model(images)
                #print(output.shape)
                loss = criterion(output,labels)
                train_loss.append(loss.item())
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                y_pred = (nn.Softmax(dim=1)(output)).argmax(dim=1)
                train_acc.append(torch.sum(y_pred == labels).item()/labels.shape[0])
                running_acc += torch.sum(y_pred == labels).item()/labels.shape[0]
                #print(y_pred.shape)
            print('Train loss: ', running_loss,
            '\ttrain accuracy: ', running_acc/len(trainloader) * 100,'%')
        torch.save(model, 'final_model.pth')
        pickle.dump(train_loss,open( "train_loss.p", "wb" ))
        pickle.dump(train_acc, open('train_acc.p', "wb"))
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="final_model.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_set = mnist()
        images = test_set.images
        labels = test_set.labels
        labels = labels.long()
        model.eval()
        with torch.no_grad():
            output = model(images)
            y_pred = (nn.Softmax(dim=1)(output)).argmax(dim=1)
        accuracy = torch.sum(y_pred == labels).item()/len(labels)*100
        print('The test accuracy is: ', accuracy,'%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    