#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import glob
import os

import cv2 as cv
import numpy as np

import args
import utils

samples_names = []
folder_base = os.path.join(args.output_folder_base, args.database_name, utils.ProcessingType.TRAIN.value)
videos_names = os.listdir(folder_base)


def resize_video_sample(video_sample, size=(64, 64)):
    resized_sample = []
    for i in range(len(video_sample)):
        sample_obj = video_sample[i]
        sample_obj = cv.resize(sample_obj, size, interpolation=cv.INTER_CUBIC)
        resized_sample.append(sample_obj)

    return resized_sample


for video_name in videos_names:
    video_samples_path = glob.glob(os.path.join(folder_base, video_name, args.samples_folder_name, '*.npy'))

    print(video_name) 
    for video_sample_path in video_samples_path:
        sample = np.load(video_sample_path)
        short_file_name = video_sample_path.split(os.sep)[-1]

        if short_file_name.find('_64.npy') == -1:
            size = (64, 64)
        else:
            continue
        resized_sample = resize_video_sample(sample, size)
        np.save(video_sample_path.replace('.npy', '_64.npy'), resized_sample)
