from tests import _PATH_DATA
import numpy as np
import torch
#import sys
#sys.path.append(_PATH_DATA)
#from data import Corrupt_MNIST

dataset = torch.load(_PATH_DATA)
assert len(dataset[0].labels.data.numpy()) == 25000
print('Train data has length 25000')
assert len(dataset[1].labels.data.numpy()) == 5000
print('Test data has length 5000')
for image in dataset[0].images.data.numpy():
	assert np.expand_dims(image,0).shape == (1, 28, 28)
print('All train datapoints are of shape (1,28,28)')
for image in dataset[1].images.data.numpy():
	assert np.expand_dims(image,0).shape == (1, 28, 28)
print('All test datapoints are of shape (1,28,28)')

classes = np.array([0,1,2,3,4,5,6,7,8,9]) 
assert np.any(np.in1d(classes,dataset[0].labels.data.numpy()))
print('All train labels are represented')
assert np.any(np.in1d(classes,dataset[1].labels.data.numpy()))
print('All test labels are represented')