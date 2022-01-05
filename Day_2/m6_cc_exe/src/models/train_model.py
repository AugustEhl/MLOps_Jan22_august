# Import packages
import argparse
import glob
import pickle
import sys

import numpy as np
# Import PyTorch
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import Dataset


# Create PyTorch Dataset Class object
class Corrupt_MNIST(Dataset):
    def __init__(self, filepath, train=True):
        # If train is True it should fetch all train data
        if train:
            print("Creating training set")
            # Setting train path
            self.train_path = filepath + "/train"
            # Generating list from path
            file_list = glob.glob(self.train_path + "*")
            print("These files will be merged: \n", file_list)
            # Looping over files and store the results
            data = []
            targets = []
            for file in file_list:
                data_load = np.load(file)
                data.append(data_load["images"])
                targets.append(data_load["labels"])
            print("This is the len of data: ", len(data))
            # Cocnatenate the results into one array and then converting to PyTorch tensors
            self.images = torch.tensor(np.concatenate(data), dtype=torch.float)
            self.labels = torch.tensor(np.concatenate(targets), dtype=torch.float)
            print("Succesfully created trainset")
        # Else load test set
        else:
            print("Creating test set")
            self.test_path = filepath + "/test.npz"
            data = np.load(self.test_path)
            self.images = torch.tensor(data["images"], dtype=torch.float)
            self.labels = torch.tensor(data["labels"], dtype=torch.float)
            print("Succesfully created testset")

    def __len__(self):
        # Get length of data
        return len(self.labels)

    def __getitem__(self, idx):
        # Create index
        label = self.labels[idx]
        img = self.images[idx]
        return img, label


# Train function with given --lr --num_epochs --batch_size --weight_decay --data
def train():
    """
    Function to train model and store training results
    Requirements:
        - Data must have been generated before executing this script

    Parameters:
        (OPTIONAL)
        --lr: learning rate (Default=0.001)
        --num_epochs: Number of training loops
        --batch_size: Batch size
        --weight_decay: Weight decay term for the optimizer
        --data: Data to be used for training

    RETURNS:
        - models/final_model.pth
        - src/visualization/train_loss.p
        - src/visualization/train_acc.p
    """
    print("Training day and night")
    # Add parser
    parser = argparse.ArgumentParser(description="Training arguments")
    # Adding arguments
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="data/processed/train.pth")
    # extracting given arguments
    args = parser.parse_args(sys.argv[1:])
    print(args)

    # Creating model object
    model = MyAwesomeModel()
    # Defining loss function
    criterion = nn.CrossEntropyLoss()
    # Defining optmizer. Here, we use Adam.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.85, 0.89),
        weight_decay=args.weight_decay,
    )
    # Load the processed data
    train_set = torch.load(args.data)
    # We use PyTorch DataLoader to iterate over the data
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    # Specify number of epochs
    num_epochs = args.num_epochs
    # Creating lists to store results
    train_loss = []
    train_acc = []
    # Training
    for i in range(num_epochs):
        print("Epoch ", i + 1, " of ", num_epochs)
        running_loss = 0
        running_acc = 0
        for images, labels in trainloader:
            # Convert labels to long format
            labels = labels.long()
            # Zero the gradients
            optimizer.zero_grad()
            # Apply model
            output = model(images)
            # calculate loss
            loss = criterion(output, labels)
            # Add to the previous losses
            running_loss += loss.item()
            # Backpropagate
            loss.backward()
            # Take another step
            optimizer.step()
            # Convert to predictions
            y_pred = (nn.Softmax(dim=1)(output)).argmax(dim=1)
            running_acc += torch.sum(y_pred == labels).item() / labels.shape[0]
        # Store results
        train_loss.append(running_loss)
        train_acc.append(running_acc / len(trainloader) * 100)
        print(
            "Train loss: ",
            running_loss,
            "\ttrain accuracy: ",
            running_acc / len(trainloader) * 100,
            "%",
        )
    # Save results
    torch.save(model, "models/final_model.pth")
    pickle.dump(train_loss, open("src/visualization/train_loss.p", "wb"))
    pickle.dump(train_acc, open("src/visualization/train_acc.p", "wb"))


# Main
if __name__ == "__main__":
    train()
