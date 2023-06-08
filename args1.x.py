#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import numpy as np
import tensorflow as tf
import os
from utils import ProcessingType, log_message, check_file_existence
import pdb
import sys
from utils import create_dir

operating_system = sys.platform

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
temporal_size = 15  # when testing set to 3
temporal_offsets = np.arange(-temporal_size, temporal_size + 1, 1)
print('temporal_offsets', temporal_offsets)
detection_threshold = 0.5

database_name = 'ped2'
output_folder_base = '/media/lili/SSD2/datasets/abnormal_event/ped2/output_yolo_%.2f' % detection_threshold
input_folder_base = '/media/lili/SSD2/datasets/abnormal_event/ped2'
samples_folder_name = 'images_%d_%.2f' % (temporal_size, detection_threshold)
samples_folder_name_context = 'images_with_context_%d_%.2f' % (temporal_size, detection_threshold)
optical_flow_folder_name = 'optical_flow_%d_%.2f' % (temporal_size, detection_threshold)
meta_folder_name = 'meta_%d_%.2f' % (temporal_size, detection_threshold)
imagenet_logits_folder_name = 'imagenet_logits_before_softmax'


def set_temporal_size(temporal_size_):
    global temporal_size
    temporal_size = temporal_size_


block_scale = 3
logs_folder = "logs"
num_samples_for_visualization = 500
CHECKPOINTS_PREFIX = 'conv3d_4_tasks_0.5_mae_wide_deep_resnet_3_obj_relu_resnet'  # 'conv3d_4_tasks_0.5_mae_wide_deep_resnet_3_5_losses_0.5_obj'  # % temporal_size #  'conv3d_context_slim_%d_2' % temporal_size


CHECKPOINTS_BASE = os.path.join(output_folder_base, database_name, "checkpoints", CHECKPOINTS_PREFIX)
create_dir(CHECKPOINTS_BASE)

allowed_video_extensions = ['avi', 'mp4']
allowed_image_extensions = ['jpg', 'png', 'jpeg']
RESTORE_FROM_HISTORY = True

history_filename = "history_%s_%s.txt" % (database_name, '%s')

if RESTORE_FROM_HISTORY is False:
    print('removing history...')
    if check_file_existence(history_filename % ProcessingType.TRAIN.value):
        os.remove(history_filename % ProcessingType.TRAIN.value)
    if check_file_existence(history_filename % ProcessingType.TEST.value):
        os.remove(history_filename % ProcessingType.TEST.value)


def log_parameters():
    message = "\n" * 5 + "Starting the algorithm with the following parameters: \n"
    local_vars = globals()
    for v in local_vars.keys():
        if not v.startswith('_'):
            message += " " * 5 + v + "=" + str(local_vars[v]) + "\n"
    log_message(message)

