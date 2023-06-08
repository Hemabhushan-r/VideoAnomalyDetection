#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import cv2 as cv
import pdb

from utils import read_graph_and_init_session
from obj_det.detection_meta import *


class MaskRCNN:
    """ Background is the first class (0 index)."""

    def __init__(self, confidence_threshold, config, is_bgr=True):
        """

        :param confidence_threshold: the confidence for detection
        :param is_bgr: if the image is in the bgr format.
        """

        self.confidence_threshold = confidence_threshold
        self.num_classes = 90
        self.graph_name = 'mask_rcnn'
        self.session = read_graph_and_init_session('./models/mask_rcnn/frozen_inference_graph.pb',
                                                   self.graph_name, config)
        self.is_bgr = is_bgr
        # self.__init_all_sessions()

    def __init_all_sessions(self):
        image = np.ones((self.network_input_size[0], self.network_input_size[0], 3))

        # detection network
        self.__run_detection_network([self.__preprocessing_image_for_detection_network(image)])

    def __run_detection_network(self, images):  
        boxes, scores, classes, num_detections, masks = self.session.run(
            [self.session.graph.get_tensor_by_name(self.graph_name + '/detection_boxes:0'),
             self.session.graph.get_tensor_by_name(self.graph_name + '/detection_scores:0'),
             self.session.graph.get_tensor_by_name(self.graph_name + '/detection_classes:0'),
             self.session.graph.get_tensor_by_name(self.graph_name + '/num_detections:0'),
             self.session.graph.get_tensor_by_name(self.graph_name + '/detection_masks:0')

             ], feed_dict={self.graph_name + '/image_tensor:0': images})
        return boxes, scores, classes, num_detections, masks

    def __preprocessing_image_for_detection_network(self, image):
        if self.is_bgr:
            image = image[:, :, [2, 1, 0]]  # BGR2RGB only for fpn
        return image

    def __preprocessing_images_for_detection_network(self, images):
        processed_images = []
        for image in images:
            processed_images.append(self.__preprocessing_image_for_detection_network(image))

        return np.array(processed_images)

    def get_detections_batch(self, images):
        processed_images = self.__preprocessing_images_for_detection_network(images)
        detection_boxes, detection_scores, detection_classes, num_detections, masks = self. \
            run_detection_network(processed_images)

        images_detections = []

        rows = images[0].shape[0]
        cols = images[0].shape[1]

        for index_detections in range(len(images)):
            frame_detections = []
            num_detections_frame = int(num_detections[index_detections])
            for detection_id in range(num_detections_frame):
                bbox = [float(v) for v in detection_boxes[index_detections][detection_id]]
                confidence = float(detection_scores[index_detections][detection_id])
                class_id = int(detection_classes[index_detections][detection_id])

                if confidence < self.confidence_threshold:
                    continue  # continue if the score is lower

                start_x = int(bbox[1] * cols)
                start_y = int(bbox[0] * rows)
                end_x = int(bbox[3] * cols)
                end_y = int(bbox[2] * rows)

                detection_meta = DetectionMeta(x_min=max(0.0, start_x),  # xmin
                                               y_min=max(0.0, start_y),  # ymin
                                               x_max=min(np.float(cols), end_x),  # xmax
                                               y_max=min(np.float(rows), end_y),  # ymax
                                               detection_confidence=confidence,
                                               class_id=class_id,
                                               mask=masks[index_detections][detection_id])
                frame_detections.append(detection_meta)

            images_detections.append(frame_detections)

        return images_detections

    def get_detections(self, image):
        processed_image = self.__preprocessing_image_for_detection_network(image)
        detection_boxes, detection_scores, detection_classes, num_detections, masks = self. \
            __run_detection_network([processed_image])

        image_detections = []

        rows = image.shape[0]
        cols = image.shape[1]

        for index_detections in range(1):
            num_detections_frame = int(num_detections[index_detections])
            for detection_id in range(num_detections_frame):
                bbox = [float(v) for v in detection_boxes[index_detections][detection_id]]
                confidence = float(detection_scores[index_detections][detection_id])
                class_id = int(detection_classes[index_detections][detection_id])
                if confidence < self.confidence_threshold:
                    continue  # continue if the score is lower

                start_x = int(bbox[1] * cols)
                start_y = int(bbox[0] * rows)
                end_x = int(bbox[3] * cols)
                end_y = int(bbox[2] * rows)

                detection_meta = DetectionMeta(x_min=max(0.0, start_x),  # xmin
                                               y_min=max(0.0, start_y),  # ymin
                                               x_max=min(np.float(cols), end_x),  # xmax
                                               y_max=min(np.float(rows), end_y),  # ymax
                                               detection_confidence=confidence,
                                               class_id=class_id,
                                               mask=masks[index_detections][detection_id])
                image_detections.append(detection_meta)

        return image_detections

