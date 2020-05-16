# define a configuration for the model
from mrcnn.config import Config
import os

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

subdirs = ['sim-2017', 'sim-2018']
train_dirs = [os.path.join('/data',y) for y in subdirs]
validation_dirs = [os.path.join('/data/validation',y) for y in subdirs]
test_dir = "/data/test"
