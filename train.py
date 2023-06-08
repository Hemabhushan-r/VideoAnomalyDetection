#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import os
import sys
operating_system = sys.platform

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if operating_system.find("win") == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import args
import datetime
import tensorflow as tf

import utils
from utils import ProcessingType, create_dir
from object_extraction import *
import middle_fbwd_consecutive_resnet.trainer as trainer
from compute_performance_scores import *

# do not delete this
RUNNING_ID = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
utils.set_vars(args.logs_folder, RUNNING_ID)
utils.create_dir(args.logs_folder)
args.log_parameters()

assert args.temporal_size == 15
# extract the objects
extract_objects(ProcessingType.TRAIN, is_video=False)

# resize objects for training
from resize_video_patches import *
from extract_features_resnet import features_extractor
features_extractor.extract()

trainer.train()

