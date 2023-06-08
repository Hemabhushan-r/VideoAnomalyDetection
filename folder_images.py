#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import os
import numpy as np
import cv2 as cv
import ntpath
import pdb

import args
from utils import get_extension, log_message, log_error


class FolderImage:
    # TODO: maybe it s better to create a base class
    def __init__(self, images_path):
        self.name = ntpath.basename(images_path)
        self.images_path = images_path
        self.frames = self.read_images()
        self.num_frames = len(self.frames)
        self.nn = 0
        log_message("There are %d images in %s" % (self.num_frames, self.images_path))

        self.has_next = True
        if self.num_frames > 0:
            self.is_valid = True
        else:
            self.is_valid = False
            log_error("There are 0 images in the folder %s." % self.images_path)
        self.width = self.frames[0].shape[1]
        self.height = self.frames[0].shape[0]
        self.fps = 25

    def read_images(self):
        images_names = os.listdir(self.images_path)
        images_names.sort()
        frames = []
        for image_name in images_names:
            if get_extension(image_name) in args.allowed_image_extensions:
                image = cv.imread(os.path.join(self.images_path, image_name))
                frames.append(image)
        return frames

    def read_frame(self):
        if len(self.frames) > 0:
            frame = self.frames.pop(0)
            return frame
        else:
            return None

    def read_all_frames(self):
        frame = self.read_frame()
        while frame is not None:
            self.frames.append(frame)
            frame = self.read_frame()
