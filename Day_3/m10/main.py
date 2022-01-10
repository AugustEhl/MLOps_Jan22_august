import numpy as np
import pickle

import torch
from torch import nn
from torch import optim

from data import data_mnist
from model_2 import MyAwesomeModel
import hydra


@hydra.main(config_path="conf",config_name="model_training_config.yaml")
def train(cfg):
    print("Training day and night")
    train_params = cfg.conf_training
    print(train_params)
    lr = train_params['learning_rate']
    num_epochs = train_params['epochs']
    batch_size = train_params['batch_size']
    weight_decay = train_params['weight_decay']
    seed = train_params['seed']

    # TODO: Implement training loop here
    model_params = cfg.conf_model
    print(model_params)
    model = MyAwesomeModel(hidden_dim=model_params['hidden_dim'], p=model_params['dropout'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.85,0.89),weight_decay=weight_decay)
    #print(optimizer)
    train_set= data_mnist
    print('Trainset: ', train_set.labels)
    torch.manual_seed(seed)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
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


if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
    