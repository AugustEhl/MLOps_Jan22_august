from tests import _PATH_MODEL
import numpy as np
from torch import nn
import torch
import sys
sys.path.append(_PATH_MODEL)
from model import MyAwesomeModel

model = MyAwesomeModel()
assert list(model.parameters())[0].size() == torch.Size([256, 784]), "Wrong input size to model"
assert list(model.parameters())[-1].size() == torch.Size([10]), "Wrong output size of model"

#def test_error_on_wrong_shape():
#   with pytest.raises(ValueError, match="Input shape must be (5, 28, 28)")
#      model(torch.randn(1,2,3))