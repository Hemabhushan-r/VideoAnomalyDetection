#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import tensorflow as tf
import numpy as np
import cv2 as cv
import pdb
import timeit
import sys

from obj_det.yolo.model import yolov3
from obj_det.detection_meta import *
from utils import log_message, log_error, check_file_existence


class YoloV3:

    def __init__(self, confidence_threshold, config, is_rgb=False):

        # input size net
        self.input_size = (416, 416)
        self.confidence_threshold = confidence_threshold
        self.num_classes = 80
        self.is_rgb = is_rgb
        self.anchors = [[10., 13.], [16., 30.], [33., 23.], [30., 61.], [62., 45.], [59., 119.], [116., 90.], [156., 198.], [373., 326.]]

        # reading: object detector
        object_detector_nn_path = "./models/yolov3/yolov3.ckpt"
        self.obj_detection_sess = self.create_session(object_detector_nn_path, config)

        self.__init_session()

    def __init_session(self):

        image = np.ones((self.input_size[0], self.input_size[0], 3))

        # detection network
        YoloV3.run_detection_network([self.preprocessing_image_for_detection_network(image)], self.obj_detection_sess)

    @staticmethod
    def run_detection_network(images, session):
        pred_boxes_, pred_confs_, pred_probs_ = session.run([session.graph.get_tensor_by_name('yolo/pred_boxes:0'),
                                                             session.graph.get_tensor_by_name('yolo/pred_confs:0'),
                                                             session.graph.get_tensor_by_name('yolo/pred_probs:0')],
                                                            feed_dict={'input_data:0': images})

        return pred_boxes_, pred_confs_, pred_probs_

    def preprocessing_image_for_detection_network(self, image, return_all=False):

        # the image must be RGB. We do not subtract any mean and divide to std,
        # because this preprocessing is inside the network.

        img, resize_ratio, dw, dh = self.letterbox_resize(image, self.input_size[0], self.input_size[1])
        if not self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = img / 255.0
        if return_all:
            return img, resize_ratio, dw, dh
        else:
            return img

    @staticmethod
    def letterbox_resize(img, new_width, new_height, interp=0):
        # Letterbox resize. keep the original aspect ratio in the resized image.
        ori_height, ori_width = img.shape[:2]

        resize_ratio = min(new_width / ori_width, new_height / ori_height)

        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)

        img = cv.resize(img, (resize_w, resize_h), interpolation=interp)
        image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)

        image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

        return image_padded, resize_ratio, dw, dh

    def preprocessing_images_for_detection_network(self, images):
        processed_images = []

        for image in images:
            processed_images.append(self.preprocessing_image_for_detection_network(image))

        return np.array(processed_images)

    def get_bboxes(self, pred_bboxes, pred_scores):
        boxes_, scores_, labels_ = self.obj_detection_sess.run([self.obj_detection_sess.graph.get_tensor_by_name('yolo/boxes:0'),
                                                                self.obj_detection_sess.graph.get_tensor_by_name('yolo/scores:0'),
                                                                self.obj_detection_sess.graph.get_tensor_by_name('yolo/label:0')],
                                                               feed_dict={'pred_boxes_ph:0': [pred_bboxes], 'pred_scores_ph:0': [pred_scores]})
        return boxes_, scores_, labels_

    def get_detections_batch(self, images):

        dummy_image = np.zeros(images[0].shape)
        processed_images = self.preprocessing_images_for_detection_network(images)
        pred_boxes_, pred_confs_, pred_probs_ = YoloV3.run_detection_network(processed_images, self.obj_detection_sess)
        pred_scores_ = pred_confs_ * pred_probs_

        _, resize_ratio, dw, dh = self.letterbox_resize(dummy_image, self.input_size[0], self.input_size[1])

        rows = images[0].shape[0]
        cols = images[0].shape[1]
        image_detections = []
        for index in range(len(processed_images)):
            frame_detections = []
            boxes_, scores_, labels_ = self.get_bboxes(pred_boxes_[index], pred_scores_[index])
            # rescale the coordinates to the original image
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

            for idx_det in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[idx_det]
                if scores_[idx_det] < self.confidence_threshold:
                    continue  # continue if the score is lower
                detection_meta = DetectionMeta(x_min=max(0.0, x0),  # xmin
                                               y_min=max(0.0, y0),  # ymin
                                               x_max=min(np.float(cols), x1),  # xmax
                                               y_max=min(np.float(rows), y1),  # ymax
                                               detection_confidence=scores_[idx_det],
                                               class_id=labels_[idx_det] + 1)
                frame_detections.append(detection_meta)

            image_detections.append(frame_detections)

        return image_detections

    def get_detections(self, image):

        dummy_image = np.zeros(image.shape)
        processed_image = self.preprocessing_image_for_detection_network(image)
        pred_boxes_, pred_confs_, pred_probs_ = YoloV3.run_detection_network([processed_image],
                                                                                  self.obj_detection_sess)
        pred_scores_ = pred_confs_ * pred_probs_

        _, resize_ratio, dw, dh = self.letterbox_resize(dummy_image, self.input_size[0], self.input_size[1])

        rows = image.shape[0]
        cols = image.shape[1]
        image_detections = []
        for index in range(1):
            boxes_, scores_, labels_ = self.get_bboxes(pred_boxes_[index], pred_scores_[index])
            # rescale the coordinates to the original image
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

            for idx_det in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[idx_det]
                if scores_[idx_det] < self.confidence_threshold:
                    continue  # continue if the score is lower
                detection_meta = DetectionMeta(x_min=max(0.0, x0),  # xmin
                                               y_min=max(0.0, y0),  # ymin
                                               x_max=min(np.float(cols), x1),  # xmax
                                               y_max=min(np.float(rows), y1),  # ymax
                                               detection_confidence=scores_[idx_det],
                                               class_id=labels_[idx_det] + 1)
                image_detections.append(detection_meta)

        return image_detections

    def close_sess(self):
        self.obj_detection_sess.close()

    def create_session(self, graph_path, config):
        if not check_file_existence(graph_path + '.meta'):
            log_error("%s is missing." % graph_path)
            sys.exit(-1)

        yolo_graph = tf.Graph()
        session = tf.Session(graph=yolo_graph, config=config)

        with yolo_graph.as_default():
            input_data = tf.placeholder(tf.float32, [None, self.input_size[1], self.input_size[0], 3], name='input_data')
            yolo_model = yolov3(self.num_classes, self.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_boxes_ph = tf.placeholder(tf.float32, [1, 10647, 4], name='pred_boxes_ph')
            pred_scores_ph = tf.placeholder(tf.float32, [1, 10647, 80], name='pred_scores_ph')

            boxes, scores, labels = YoloV3.gpu_nms(pred_boxes_ph, pred_scores_ph, self.num_classes, max_boxes=200, score_thresh=0.1, nms_thresh=0.45)

            saver = tf.train.Saver()
            saver.restore(session, graph_path)

        return session

    @staticmethod
    def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
        """
        Perform NMS on GPU using TensorFlow.
        params:
            boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
            scores: tensor of shape [1, 10647, num_classes], score=conf*prob
            num_classes: total number of classes
            max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
            score_thresh: if [ highest class probability score < score_threshold]
                            then get rid of the corresponding box
            nms_thresh: real value, "intersection over union" threshold used for NMS filtering
        """

        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # since we do nms for single image, then reshape it
        boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
        score = tf.reshape(scores, [-1, num_classes])

        # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
        mask = tf.greater_equal(score, tf.constant(score_thresh))
        # Step 2: Do non_max_suppression for each class
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:, i])
            filter_score = tf.boolean_mask(score[:, i], mask[:, i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        boxes = tf.concat(boxes_list, axis=0, name="yolo/boxes")
        score = tf.concat(score_list, axis=0, name="yolo/scores")
        label = tf.concat(label_list, axis=0, name="yolo/label")

        return boxes, score, label
