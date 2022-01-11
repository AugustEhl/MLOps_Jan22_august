from tests import _PATH_TRAINING
from tests import _PATH_DATA
import sys
sys.path.append(_PATH_TRAINING)
sys.path.append(_PATH_DATA)
from training import num_epochs, lr, batch_size, weight_decay

assert num_epochs != None
assert lr != None
assert batch_size != None
assert weight_decay != None


