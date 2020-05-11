# define a configuration for the model
from mrcnn.config import Config

class DeepVisionConfig(Config):
    NAME = "deepvision_cfg"
    NUM_CLASSES = 4 + 1  # herring, blue whiting, mackerel, lanternfish
    STEPS_PER_EPOCH = 1000
    # more parameters...

class_names = ['BG','bluewhiting','herring','lanternfish','mackerel']
