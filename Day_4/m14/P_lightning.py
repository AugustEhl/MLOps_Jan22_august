# PyTorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model import MyAwesomeModel
from data import mnist
import torch
import os
import wandb

train_set, _ = mnist('../../../dtu_mlops/data/corruptmnist/')

def train():
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=65, shuffle=True)
    model = MyAwesomeModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="train_loss", mode="min"
    )
    trainer = Trainer(accelerator='cpu',
        max_epochs = 3,
        default_root_dir=os.getcwd(),
        limit_train_batches=0.2,
        callbacks=[checkpoint_callback],
        logger=WandbLogger(project="dtu_mlops_mnist")
    )
    trainer.fit(model,train_dataloader=train_loader)

train()
    

