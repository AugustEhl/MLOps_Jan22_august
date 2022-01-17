"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import glob
import os
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        self.img_path = path_to_folder
        self.actors = os.listdir(self.img_path)
        
    def __len__(self):
        return len(self.actors)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        actor = self.actors[index]
        #print(actor)
        act_img = glob.glob(self.img_path + '/' + actor + '/*.jpg')
        #print(act_img)
        img = Image.open(act_img[0])
        #print(img)
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw-deepfunneled', type=str)
    parser.add_argument('-num_workers', default=8, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        batch = next(iter(dataloader))
        print(batch.shape)
        images = make_grid([img for img in batch])
        show(images)
        
        
    if args.get_timing:
        # lets do so repetitions
        res_mean = []
        res_std = []
        for num_work in range(1,9):
            print('Num_workers: ', num_work)
            dataloader = DataLoader(dataset, batch_size=250, shuffle=False,
                            num_workers=num_work)
            res = [ ]
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    print(f'Batch: {batch_idx+1} of {len(dataloader)}')
                    if batch_idx > 100:
                        break
                end = time.time()

                res.append(end - start)
            
            res = np.array(res)
            res_mean.append(np.mean(res))
            res_std.append(np.std(res))
            print(f'Timing: {np.mean(res)}+-{np.std(res)}')
        plt.errorbar(range(1,9), res_mean, yerr=res_std)
        plt.title('Error-plot')
        plt.xlabel('Num_workers')
        plt.ylabel('Error')
        plt.savefig('error_plot.jpg')
        plt.close()
