# Import packages
import argparse
import glob
import sys
from copy import deepcopy
from sklearn.manifold import Isomap

import numpy as np
# Import PyTorch
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import Dataset
import torchdrift
import torch.nn.functional as F
import matplotlib.pyplot as plt

def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

# Create PyTorch Dataset Class object
class Corrupt_MNIST(Dataset):
    def __init__(self, filepath, train=True, Transform=None):
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
            if Transform:
                self.images = Transform(self.images)
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
        img = torch.unsqueeze(self.images[idx],0)
        #print(img.shape)
        return img, label


# Train function with given --lr --num_epochs --batch_size --weight_decay --data
def model_drift():
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
    parser.add_argument("--model", type=str, default="final_model.pth")
    parser.add_argument("--data", type=str, default="../../Day_2/m6_cc_exe/data/raw")
    # extracting given arguments
    args = parser.parse_args(sys.argv[1:])
    print(args)

    # Creating model object
    model = torch.load(args.model).eval()

    # Load the processed data
    train_set = Corrupt_MNIST(args.data,train=True)
    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=65,
        shuffle=False
    )
    N=6
    batch_train,_ = next(iter(trainloader))
    batch_ood = corruption_function(batch_train)
    inps = torch.cat([batch_train[:N], batch_ood[:N]])
    predictions = F.softmax(model(inps), dim=1).argmax(dim=1).data.numpy()
    y_preds = [["0", "1", "2", "3", "4", "5", "6", "7", "8", "0"][i] for i in predictions]
    plt.figure(figsize=(15, 5))
    for i in range(2 * N):
        plt.subplot(2, N, i + 1)
        plt.title(y_preds[i])
        plt.imshow(inps[i].permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("drift_plots.jpg")
    plt.close()
    
    feature_extractor = deepcopy(model)
    features = feature_extractor(batch_train[:10])
    #feature_extractor = torch.nn.Identity()
    drift_detectors = [torchdrift.detectors.mmd.RationalQuadraticKernel(),
        torchdrift.detectors.mmd.ExpKernel(),
        torchdrift.detectors.mmd.GaussianKernel()    
    ]
    Results = np.array([])
    for dd in drift_detectors:
        drift_detector = torchdrift.detectors.KernelMMDDriftDetector(kernel=dd)
        torchdrift.utils.fit(trainloader, feature_extractor, drift_detector)
        score = drift_detector(features)
        p_val = drift_detector.compute_p_value(features)
        Results = np.hstack((Results, np.array([score,p_val])))
    print(Results)
    #drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    #torchdrift.utils.fit(trainloader, feature_extractor, drift_detector)
    #drift_detection_model = torch.nn.Sequential(
    #feature_extractor,
    #drift_detector
    #)
    #score = drift_detector(features)
    #p_val = drift_detector.compute_p_value(features)
    #score, p_val
    #N_base = drift_detector.base_outputs.size(0)
    #mapper = Isomap(n_components=2)
    #base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    #features_embedded = mapper.transform(features)
    #plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    #plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    #plt.title(f'score {score:.2f} p-value {p_val:.2f}')
    #plt.savefig("Isomap_base.jpg")
    #plt.close()
    #features = feature_extractor(batch_ood)
    #score = drift_detector(features)
    #p_val = drift_detector.compute_p_value(features)

    #features_embedded = mapper.transform(features)
    #plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    #plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    #plt.title(f'score {score:.2f} p-value {p_val:.2f}')
    #plt.savefig("Isomap_corrupted.jpg")
    #plt.close()


# Main
if __name__ == "__main__":
    model_drift()
