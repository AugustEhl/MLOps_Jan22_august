from torch import nn, optim
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
import wandb


class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Sequential(nn.Linear(784,256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2))
        self.hidden2 = nn.Sequential(nn.Linear(256,128),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2))
        self.criterium = nn.CrossEntropyLoss()
        # Output layer, 10 units - one for each digit
        self.fc1 = nn.Sequential(nn.Linear(128,10))
        self.table = wandb.Table(columns=["Height"])
        
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        # 1. Hidden layer
        x = self.hidden1(x)
        # 2. Hidden layer
        x = self.hidden2(x)
        # Output layer with softmax activation
        return self.fc1(x)
    
    def training_step(self, batch, batch_idx):
        data, targets = batch
        preds = self(data)
        loss = self.criterium(preds,targets)
        acc = (targets == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        predictions = preds.argmax(dim=-1)
        pred_targets = [[s.item()] for s in predictions]
        self.table.add_data(pred_targets)
        self.logger.experiment.log({'logits': wandb.plot.histogram(self.table,"Height",title='Predicted target distribution')})
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.85,0.89),
            weight_decay=1e-3
        )