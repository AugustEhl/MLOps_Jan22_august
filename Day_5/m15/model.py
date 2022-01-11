from torch import nn, save
import torch

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Sequential(nn.Linear(784,256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2))
        self.hidden2 = nn.Sequential(nn.Linear(256,128),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2))
        # Output layer, 10 units - one for each digit
        self.fc1 = nn.Sequential(nn.Linear(128,10))
        
    def forward(self, x):
        #print(x.shape)
        if x.shape[0] != 5 or x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError('Input shape must be (5, 1, 28, 28)')
        x = x.view(x.shape[0],-1)
        if x.shape != torch.Size([x.shape[0], 784]):
            raise ValueError('x must be flattened to size (batch_size,784)')
        # 1. Hidden layer
        x = self.hidden1(x)
        # 2. Hidden layer
        x = self.hidden2(x)
        # Output layer with softmax activation
        return self.fc1(x)