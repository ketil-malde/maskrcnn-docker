# define a configuration for the model
from mrcnn.config import Config

class DeepVisionConfig(Config):
    NAME = "deepvision_cfg"
    NUM_CLASSES = 4 + 1  # herring, blue whiting, mackerel, lanternfish
    STEPS_PER_EPOCH = 1000
    # more parameters...

class_names = ['BG','bluewhiting','herring','lanternfish','mackerel']

# layers to train.  Can be one of 'all', '3+', '4+', 'heads'
train_layers = '3+'
initial_weights = 'mask_rcnn_coco.h5'
epochs = 10

# Print with color to distinguish from all the text barf
def pr(*strng):
    print('\033[93m', end='')
    for s in strng: print(s, end=' ')
    print('\033[0m')
