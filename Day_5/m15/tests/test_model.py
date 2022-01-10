from tests import _PATH_MODEL
import numpy as np
from torch import nn
import torch
import sys
sys.path.append(_PATH_MODEL)
from model import MyAwesomeModel

model = MyAwesomeModel()
assert list(model.parameters())[0].size() == torch.Size([256, 784])
assert list(model.parameters())[-1].size() == torch.Size([10])




