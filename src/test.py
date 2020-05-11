# rip from https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import os

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

if not os.path.exists('mask_rcnn_coco.h5'):
    print('No weights found (use download_weights.py)')
    exit()

os.mkdir('out')
    
# Directory of images to run detection on
IMAGE_DIR = "/data/test"
WEIGHTS   = "current.h5"

from config import DeepVisionConfig, class_names
config = DeepVisionConfig()
config.BATCH_SIZE = 1
config.IMAGES_PER_GPU = 1
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='models', config=config)

# Load weights trained on MS-COCO
model.load_weights(WEIGHTS, by_name=True)

# Load a random image from the images folder
for f in os.listdir(IMAGE_DIR):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, f))
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
    plt.savefig(os.path.join('out',f))

