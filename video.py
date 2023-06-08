#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import cv2 as cv
import numpy as np
import os

import utils


class Video:

    def __init__(self, video_path):
        file_short_name, file_extension = os.path.splitext(video_path[video_path.rfind(os.path.sep) + 1:len(video_path)])
        self.name = file_short_name

        video_extension_and_fourcc_avi = ('.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        video_extension_and_fourcc_mp4 = ('.mp4', 0x7634706d)

        if file_extension == ".avi":
            self.video_extension_and_fourcc = video_extension_and_fourcc_avi
        elif file_extension == ".mp4":
            self.video_extension_and_fourcc = video_extension_and_fourcc_mp4

        self.num_frames = 0
        self.has_next = True
        self.is_valid = True

        self.capture = cv.VideoCapture(video_path)
        if self.capture.isOpened() is False:
            utils.log_error("Could not read video file %s." % video_path)
            self.is_valid = False

        self.fps = round(self.capture.get(cv.CAP_PROP_FPS))
        self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.last_frames_read = None
        self.frames = []  # use it only if u need

        if self.width == 0 or self.height == 0:
            utils.log_error("Could not read video file %s." % video_path)
            self.is_valid = False

    def get_next_frame(self, n_frames):
        if self.has_next is False:
            return None

        frames = []
        while n_frames > 0 and self.has_next and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret is False:
                self.has_next = False
                frames = None
            else:
                self.num_frames += 1
                frames.append(frame)

            n_frames -= 1

        if self.has_next is False:
            self.capture.release()

        if frames is None:
            return None

        if self.last_frames_read is not None:
            self.last_frames_read += frames
        else:
            self.last_frames_read = frames

        return self.last_frames_read[-1].copy()

    def read(self, num_frames_to_read):

        if self.last_frames_read is not None:
            frames = self.last_frames_read
            num_frames_to_read -= len(frames)
            self.last_frames_read = None
        else:
            frames = []

        while self.has_next and self.capture.isOpened() and num_frames_to_read > 0:

            ret, frame = self.capture.read()

            if ret is True:
                frames.append(frame)
                self.num_frames += 1
                num_frames_to_read -= 1
            else:
                self.has_next = False
                break

        if self.has_next is False:
            self.capture.release()

        return np.array(frames, np.float32)

    def read_all_frames(self):

        while self.has_next and self.capture.isOpened():
            ret, frame = self.capture.read()

            if ret is True:
                self.frames.append(frame)
                self.num_frames += 1
            else:
                self.has_next = False
                break

        if self.has_next is False:
            self.capture.release()

    def read_frame(self):
        frame = self.read(1)
        if len(frame) > 0:
            return frame[0]
        else:
            return None

