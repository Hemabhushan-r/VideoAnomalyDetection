#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import os
import args
import pdb
import cv2 as cv
import glob

import utils


class DataSetReader:
    def __init__(self, sample_names):
        self.sample_names = sample_names
        self.sample_names = shuffle(self.sample_names)
        self.len_seq = 7
        self.num_samples = len(self.sample_names)
        self.end_index = 0
        self.middle_frame_idx = 3
        self.negative_permutation = np.arange(self.len_seq)[::-1]

    def generate_positions(self):
        positions = []
        num_pos = 6
        for _ in range(num_pos):
            pos = np.random.randint(1, 4)
            positions.append(pos)

        num_left = int(num_pos / 2)
        left_part = np.array(positions[:num_left]) * -1

        for i in range(num_left - 1):
            left_part[i] = left_part[i] + np.sum(left_part[i + 1:])

        right_part = positions[num_left:]
        for i in range(1, num_left):
            right_part[i] += right_part[i - 1]

        new_pos = np.array(list(left_part) + [0] + right_part) + 15

        return new_pos

    def read_samples(self, files_path):
        samples = []
        samples_consecutive = []
        labels_detector = []
        labels_fwd_bwd = []
        labels_consecutive = []

        samples_resnet = []
        labels_resnet = []

        for file_path in files_path:
            full_sample_cons = np.load(file_path)

            full_sample = full_sample_cons[12:19]

            samples_resnet.append(full_sample)
            # load label
            label_resnet = np.load(file_path.replace(args.samples_folder_name, args.imagenet_logits_folder_name).replace('_64.npy', '.npy'))

            meta = np.loadtxt(file_path.replace(args.samples_folder_name, args.meta_folder_name).replace('_64.npy', '.txt'))
            yolo_logits = np.zeros(80)
            yolo_logits[int(meta[-2]) - 1] = meta[-1]
            logits_resnet = np.maximum(np.squeeze(label_resnet), 0)

            labels_resnet.append(np.concatenate((logits_resnet, yolo_logits)))

            if np.random.rand() >= 0.5:
                labels_fwd_bwd.append([1, 0])
            else:
                labels_fwd_bwd.append([0, 1])
                full_sample = full_sample[self.negative_permutation]

            if np.random.rand() >= 0.5:
                labels_consecutive.append([1, 0])
                samples_consecutive.append(full_sample)
            else:
                positions = self.generate_positions()
                sample_ = full_sample_cons[positions]
                samples_consecutive.append(sample_)
                labels_consecutive.append([0, 1])

            label = full_sample[self.middle_frame_idx]
            sample = np.delete(full_sample, self.middle_frame_idx, 0)

            samples.append(sample)
            labels_detector.append(label)

        return samples, samples_consecutive, samples_resnet,  labels_detector, labels_fwd_bwd, labels_consecutive, labels_resnet

    def next_batch(self, bach_size=256):

        if self.end_index == self.num_samples:
            self.end_index = 0
            self.sample_names = shuffle(self.sample_names)

        start_index = self.end_index
        self.end_index += bach_size
        self.end_index = min(self.end_index, self.num_samples)
        
        names = self.sample_names[start_index:self.end_index]
        samples, samples_consecutive, samples_resnet,\
        labels_detector, labels_fwd_bwd, labels_consecutive, labels_resnet = self.read_samples(names)
        samples = np.array(samples) / 255.0
        samples_consecutive = np.array(samples_consecutive) / 255.0
        samples_resnet = np.array(samples_resnet) / 255.0

        labels_detector = np.array(labels_detector) / 255.0
        labels_fwd_bwd = np.array(labels_fwd_bwd)
        labels_consecutive = np.array(labels_consecutive)
        return samples, samples_consecutive, samples_resnet, labels_detector, labels_fwd_bwd, labels_consecutive, labels_resnet


def create_readers_split():
    folder_base = os.path.join(args.output_folder_base, args.database_name, utils.ProcessingType.TRAIN.value)
    videos_names = os.listdir(folder_base)
    names_train = []
    names_val = []
    for video_name in videos_names:
        video_samples = glob.glob(os.path.join(folder_base, video_name, args.samples_folder_name, '*_64.npy'))
        video_samples.sort()
        num_examples = len(video_samples)
        num_training = int(0.85 * num_examples)
        names_train += video_samples[:num_training]
        names_val += video_samples[num_training:]

    print('num training examples', len(names_train))
    print('num validation examples', len(names_val))

    reader_train = DataSetReader(names_train)
    reader_val = DataSetReader(names_val)

    return reader_train, reader_val

