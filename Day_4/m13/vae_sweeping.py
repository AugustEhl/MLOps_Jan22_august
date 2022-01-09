import wandb
import pprint

sweep_config = {
    'method': 'random'
}
metric = {
    'name': 'loss',
    'goal': 'minimize'   
}
sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'batch_size': {
        'values': [128, 256, 512]
        },
}
sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="will-i-succeed-now")

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(config=None):
    # Initialize a new wandb run
    print('Initializing training')
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        loader = build_dataset(config.batch_size)
        network = build_network()
        optimizer = build_optimizer(network, config.optimizer, 1e-3)
        num_epochs = 1
        for epoch in range(num_epochs):
            print('Epoch', epoch+1, ' of ', num_epochs)
            avg_loss = train_epoch(network, loader, optimizer,config.batch_size)
            print('Avg loss: ',avg_loss)
            wandb.log({"loss": avg_loss})

def build_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    # download MNIST training dataset
    dataset = datasets.MNIST('datasets', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

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

def build_network(x_dim=784, hidden_dim=400,latent_dim=20):
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
    network = Model(Encoder=encoder, Decoder=decoder)
    return network.to(device)

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train_epoch(network, loader, optimizer,batchsize):
    cumu_loss = 0
    for batch_idx, (x, _) in enumerate(loader):
        x = x.view(x.shape[0],-1).to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        x_hat, mean, log_var = network(x)
        loss = loss_function(x, x_hat, mean, log_var)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})
    
    return cumu_loss /(batch_idx*batchsize)

wandb.agent(sweep_id, train)

