import numpy as np
from tests import _PATH_TRAINING
import sys
sys.path.append(_PATH_TRAINING)
from training import num_epochs, lr, batch_size, weight_decay

assert num_epochs != None
assert lr != None
assert batch_size != None
assert weight_decay != None


