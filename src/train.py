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

from config import DeepVisionConfig, pr
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
                 image_id = filename[:-4]  # skip .png suffix
                 # skip all images after 150 if we are building the train set
                 # if is_train and int(image_id) >= 150:
                 #    continue
                 # skip all images before 150 if we are building the test/val set
                 # if not is_train and int(image_id) < 150:
                 #    continue
                 img_path = os.path.join(images_dir,filename)
                 ann_path = os.path.join(images_dir,image_id + '.txt')
                 self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

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
            masks[:,:,i] = asarray(img.split()[0])/255  # alpha channel?  Nooo!

        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

subdirs = ['sim-2017', 'sim-2018']

# prepare train set
train_set = DeepVisionDataset()
train_set.load_dataset([os.path.join('/data',y) for y in subdirs], is_train=True)
train_set.prepare()
pr('Train: %d' % len(train_set.image_ids))

# prepare test/val set
test_set = DeepVisionDataset()
test_set.load_dataset([os.path.join('/data/validation',y) for y in subdirs], is_train=False)
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
    weights = model.find_last()
except FileNotFoundError:
    pr('Using initial weights from', C.initial_weights)
    if os.path.isfile('train.log'):
        pr('Backing up training log')
        os.rename('train.log', 'train.log.old')
    model.load_weights(C.initial_weights, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    eps = C.epochs
    pr('Training',eps,'epochs.')
else:
    old_epochs = int(re.findall('\d+', weights)[-2]) # end is .h5
    pr('Using weights from: ', weights)
    model.load_weights(weights, by_name=True)
    eps = old_epochs + C.epochs
    pr('Training from epoch',old_epochs,'to epoch',eps)

model.train(train_set, test_set, custom_callbacks=[logger], learning_rate=config.LEARNING_RATE, epochs=eps, layers=C.train_layers)

