#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import numpy as np
import pdb


class DetectionMeta:

    def __init__(self, x_min, y_min, x_max, y_max, detection_confidence, class_id, mask=None):

        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        self.detection_confidence = detection_confidence
        self.class_id = class_id
        self.mask = mask
        if self.mask is not None:
            self.mask[self.mask < 0.5] = 0
            self.mask[self.mask > 0.5] = 255
            self.mask = np.uint8(self.mask)

    def get_meta(self, frame_idx):
        return np.array([frame_idx, self.x_min, self.y_min, self.x_max, self.y_max, self.class_id,
                         self.detection_confidence])

    def get_width(self):
        return self.x_max - self.x_min

    def get_height(self):
        return self.y_max - self.y_min

    def get_bbox_as_array(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

    def get_bbox_as_list(self):
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def set_bbox(self, bbox):
        self.x_min = int(bbox[0])
        self.y_min = int(bbox[1])
        self.x_max = int(bbox[2])
        self.y_max = int(bbox[3])

    def set_detection_score(self, score):
        self.detection_confidence = score

    def __str__(self):
        # Override to print a readable string presentation of your object
        # below is a dynamic way of doing this without explicity constructing the string manually
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])