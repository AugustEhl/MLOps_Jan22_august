import os
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
print('_TEST_ROOT',_TEST_ROOT)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
print('_PROJECT_ROOT', _PROJECT_ROOT)
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data.pth")  # root of data
print('_PATH_DATA', _PATH_DATA)
_PATH_MODEL = os.path.join(_PROJECT_ROOT, "model.py")
print('_PATH_MODEL', _PATH_MODEL)
_PATH_TRAINING = os.path.join(_PROJECT_ROOT,"{training.py,data.pth}")
print('_PATH_TRAINING', _PATH_TRAINING)