#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import numpy as np
import os
import pdb
import scipy.io
import sys
sys.path.append("..")
from utils import get_file_name, ProcessingType
import args


mat = scipy.io.loadmat('ped2.mat')
gt = mat['gt']
output_folder = os.path.join(args.output_folder_base, args.database_name, ProcessingType.TEST.value, "%s",
                             "ground_truth_frame_level.txt")

video_names = ['%02d' % i for i in range(1, 13)]


for idx, file_name in enumerate(video_names):

    # take the num max of frames
    video_dir = '/media/lili/SSD2/datasets/abnormal_event/ped2/test/frames/' + file_name # os.path.join(args.input_folder_base, ProcessingType.TEST.value, "frames", file_name)
    print(video_dir)
    video_names = [f for f in os.listdir(video_dir)]
    print(len(video_names))
    new_content = np.zeros((len(video_names)))
    start_anomaly = gt[0][idx][0][0] - 1
    end_anomaly = gt[0][idx][1][0] 
    new_content[start_anomaly: end_anomaly] = 1
    np.savetxt(output_folder % file_name, new_content) 
    print(output_folder % file_name)
