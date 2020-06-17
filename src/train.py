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

from util import pr, find_last
from config import DeepVisionConfig
import config as C # class_names, train_layers, initial_weights

# class that defines and loads the data set
class DeepVisionDataset(Dataset):

    def load_dataset(self, dataset_dirs, is_train=True):
         for i,n in enumerate(C.class_names[1:]):
             self.add_class("dataset", i, n)

         if isinstance(dataset_dirs, str):
             images_dirs = [dataset_dirs]
         elif isinstance(dataset_dirs, list):
             images_dirs = dataset_dirs
         else:
             error("'dataset_dirs' must be a list or a string")

         for images_dir in images_dirs:
             filenames = [f for f in os.listdir(images_dir) if re.match(r'sim-201[78]_[0-9]+\.png', f)]
             pr("*** Directory: ",images_dir,"Number of images seen: ", len(filenames), "***")
             for filename in filenames:
                 self.add_image('dataset',
                                image_id = filename[:-4]  # skip .png suffix
                                path = os.path.join(images_dir,filename)
                                annotation = os.path.join(images_dir,image_id + '.txt')
                 )

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
            maskpath = os.path.join(os.path.dirname(imgpath),'mask_'+os.path.basename(imgpath)[:-4]+'_'+str(i)+'.png')
            m = Image.open(maskpath)
            masks[:,:,i] = asarray(m.split()[0])/255  # alpha channel?  Nooo!

        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# prepare train set
train_set = DeepVisionDataset()
train_set.load_dataset(C.train_dirs, is_train=True)
train_set.prepare()
pr('Train: %d' % len(train_set.image_ids))

# prepare test/val set
test_set = DeepVisionDataset()
test_set.load_dataset(C.validation_dirs, is_train=False)
test_set.prepare()

pr('Test: %d' % len(test_set.image_ids))
# prepare config
config = DeepVisionConfig()
config.display()

from tensorflow.keras.callbacks import CSVLogger
logger = CSVLogger("train.log", append=True, separator='\t')

# define the model, load weights and run training
model = MaskRCNN(mode='training', model_dir='./', config=config)

try:
    weights, old_epochs = find_last(model)
except FileNotFoundError:
    pr('Using initial weights from', C.initial_weights)
    if os.path.isfile('train.log'):
        pr('Backing up training log')
        os.rename('train.log', 'train.log.old')
    model.load_weights(C.initial_weights, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    eps = C.epochs
    pr('Training',eps,'epochs.')
else:
    pr('Using weights from: ', weights)
    model.load_weights(weights, by_name=True)
    eps = int(old_epochs) + C.epochs
    pr('Training from epoch',old_epochs,'to epoch',eps)

model.train(train_set, test_set, custom_callbacks=[logger], learning_rate=config.LEARNING_RATE, epochs=eps, layers=C.train_layers)

