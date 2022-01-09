"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import wandb
import numpy as np

sweep_config = {
    'method': 'random'
}

parameters_dict = {
    'optimizer': {
        'values': ['adam','sgd']
        },
    'batch_size_train': {
          'values': [35, 45, 65]
        }
}

sweep_config['parameters'] = parameters_dict

import pprint

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="Wandb_example_sweep")

# Model Hyperparameters
dataset_path = 'datasets'
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
#batch_size = 100
x_dim  = 784
hidden_dim = 400
latent_dim = 20
lr = 1e-3
epochs = 5


# Data loading
mnist_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(dataset_path, train=True, transform=mnist_transform, download=True)
test_dataset  = MNIST(dataset_path, train=False, transform=mnist_transform, download=True)

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

class Encoder(nn.Module):  
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.training = True
        
    def forward(self, x):
        h_       = torch.relu(self.FC_input(x))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     
                                                      
        std      = torch.exp(0.5*log_var)             
        z        = self.reparameterization(mean, std)
        
        return z, mean, log_var
       
    def reparameterization(self, mean, std,):
        epsilon = torch.randn_like(std)
        
        z = mean + std*epsilon
        
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h     = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

from torch.optim import Adam, sgd

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

#optimizer = Adam(model.parameters(), lr=lr)

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = sgd(network.parameters(),
                              lr=lr,
                              momentum=0.9)
    elif optimizer == "adam":
        optimizer = Adam(network.parameters(),
                               lr=lr)
    return optimizer

table = wandb.Table(columns=["Real Image", "Reconstructed Image"])
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        optimizer = build_optimizer(model, config.optimizer, lr)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size_train, shuffle=True)
        print("Start training VAE...")
        model.train()
        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(config.batch_size_train, x_dim)
                x = x.to(DEVICE)

                optimizer.zero_grad()

                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)
                
                overall_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                if batch_idx % 25 == 0:
                    random_img = np.random.randint(x.shape[0])
                    table.add_data(
                        wandb.Image(x.view(config.batch_size_train,1,28,28)[random_img]),
                        wandb.Image(x_hat.view(config.batch_size_train,1,28,28)[random_img])
                    )
            wandb.log({'Average Loss': overall_loss / (batch_idx*config.batch_size_train),
                'Images': table}
            )
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size_train))    
        print("Finish!!")

wandb.agent(sweep_id, train, count=5)
