# Copied from https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/
# fit a mask rcnn on the kangaroo dataset

import os
import re
import csv
from PIL import Image
from numpy import zeros, asarray

from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# TODO:
# - dataset dir etc in separate config.py
# - train and val split is parameter
# - moron mask must be improved

# define a configuration for the model
class DeepVisionConfig(Config):
    NAME = "deepvision_cfg"
    NUM_CLASSES = 4 + 1  # herring, blue whiting, mackerel, lanternfish
    STEPS_PER_EPOCH = 131
    # more parameters...

# class that defines and loads the data set
class DeepVisionDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
         self.add_class("dataset", 1, "bluewhiting")
         self.add_class("dataset", 2, "herring")         
         self.add_class("dataset", 3, "lanternfish")
         self.add_class("dataset", 4, "mackerel")
         
         images_dir = dataset_dir       # + '/images/'
         annotations_dir = dataset_dir  # + '/annots/'

         filenames = [f for f in os.listdir(images_dir) if re.match(r'simulated_images_[0-9]+\.png', f)]
         print("*** Number of images seen: ", len(filenames), "***")
         for filename in filenames:
             image_id = filename[:-4]  # skip .png suffix
             # skip all images after 150 if we are building the train set
             # if is_train and int(image_id) >= 150:
             #    continue
             # skip all images before 150 if we are building the test/val set
             # if not is_train and int(image_id) < 150:
             #    continue
             img_path = os.path.join(images_dir,filename)
             ann_path = os.path.join(annotations_dir,image_id + '.txt')
             self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file (used by load_mask below)
    # this function just constructs a rectangle corresponding to the bbox!
    # parses the Kangaroo xml file to get data
    # def extract_boxes(self, filename):
    #     # load and parse the file
    #     tree = ElementTree.parse(filename)
    #     # get the root of the document
    #     root = tree.getroot()
    #     # extract each bounding box
    #     boxes = list()
    #     for box in root.findall('.//bndbox'):
    #         xmin = int(box.find('xmin').text)
    #         ymin = int(box.find('ymin').text)
    #         xmax = int(box.find('xmax').text)
    #         ymax = int(box.find('ymax').text)
    #         coors = [xmin, ymin, xmax, ymax]
    #         boxes.append(coors)
    #     # extract image dimensions
    #     width = int(root.find('.//size/width').text)
    #     height = int(root.find('.//size/height').text)
    #     return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # create one array for all masks, each on a different channel
        annopath = self.image_info[image_id]['annotation'] 
        imgpath  = self.image_info[image_id]['path']
        img = Image.open(imgpath)

        lines = list()
        with open(annopath) as f:
            for line in csv.reader(f):
                lines.append(line)

        w, h = img.size
        masks = zeros([h, w, len(lines)], dtype='uint8')

        class_ids = list()
        for i, [p,xmin,ymin,xmax,ymax,classname] in enumerate(lines):
            class_ids.append(self.class_names.index(classname))
            img = Image.open(imgpath[:-4]+'_mask_'+str(i)+'.png')
            masks[:,:,i] = asarray(img.split()[-1])/255

        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# prepare train set
train_set = DeepVisionDataset()
train_set.load_dataset('simulated_images', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare test/val set
test_set = DeepVisionDataset()
test_set.load_dataset('simulated_images', is_train=False)
test_set.prepare()

print('Test: %d' % len(test_set.image_ids))
# prepare config
config = DeepVisionConfig()
config.display()

# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

