# rip from https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import os
import csv

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

from util import pr, find_last
from config import DeepVisionConfig, class_names
import config as C

# Directory of images to run detection on

conf = DeepVisionConfig()
conf.BATCH_SIZE = 1
conf.IMAGES_PER_GPU = 1
conf.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='./', config=conf)
weights, last_epoch = find_last(model)
out_dir = os.path.join(os.path.dirname(weights),'test_output_'+last_epoch)

pr('    Using weights from: ', weights)
pr('    Test images from:', C.test_dirs)
pr('    Writing output to: ', out_dir)

os.mkdir(out_dir)

# Load weights trained on MS-COCO
model.load_weights(weights, by_name=True)

for d in C.test_dirs:
    for root, dirs, files in os.walk(d):
        for f in files:
            try:
                image = skimage.io.imread(os.path.join(root, f))
                # Run detection
                results = model.detect([image], verbose=1)
                # Visualize results
                r = results[0]
                with open(os.path.join(out_dir,f[:-4]+'.txt')) as ofile:
                    for i, (y1, x1, y2, x2) in zip(r['class_ids'],r['rois']):
                        # note flipped x and y vs imagesim dataset defaults
                        csv.write(f,y1,x1,y2,x2,C.class_names[i])

                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                            class_names, r['scores'])
                plt.savefig(os.path.join(out_dir,f))
            except:
                pr('    Ignoring file: '+root+' '+f)

