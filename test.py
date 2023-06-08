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

from compute_performance_scores import compute_performance_indices, compute_anomaly_scores_per_object
from utils import ProcessingType
import utils
import datetime
import args

# do not delete this
RUNNING_ID = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
utils.set_vars(args.logs_folder, RUNNING_ID)
utils.create_dir(args.logs_folder)
args.log_parameters()

assert args.temporal_size == 3
extract_objects(ProcessingType.TEST, is_video=False)
compute_anomaly_scores_per_object(ProcessingType.TEST)
compute_performance_indices(ProcessingType.TEST)
