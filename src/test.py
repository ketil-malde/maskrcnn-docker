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

from config import DeepVisionConfig, class_names, pr
import config as C

if not os.path.exists('mask_rcnn_coco.h5'):
    pr('No weights found (use download_weights.py)')
    exit()

# Directory of images to run detection on
IMAGE_DIR = "/data/test"

conf = DeepVisionConfig()
conf.BATCH_SIZE = 1
conf.IMAGES_PER_GPU = 1
conf.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='./', config=conf)
weights = model.find_last()
out_dir = os.path.join(os.path.dirname(weights),'test_output')
pr('    Using weights from: ', weights)
pr('    Test images from:', IMAGE_DIR)
pr('    Writing output to: ', out_dir)

os.mkdir(out_dir)

# Load weights trained on MS-COCO
model.load_weights(weights, by_name=True)
for f in os.listdir(IMAGE_DIR):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, f))
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
    plt.savefig(os.path.join(out_dir,f))

