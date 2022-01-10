from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self,hidden_dim,p):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Sequential(nn.Linear(784,hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=p))
        self.hidden2 = nn.Sequential(nn.Linear(hidden_dim,128),
                                    nn.ReLU(),
                                    nn.Dropout(p=p))
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