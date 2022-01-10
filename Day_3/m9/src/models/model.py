from torch import nn


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
        x = x.view(x.shape[0],-1)
        # 1. Hidden layer
        x = self.hidden1(x)
        # 2. Hidden layer
        x = self.hidden2(x)
        # Output layer with softmax activation
        return self.fc1(x)