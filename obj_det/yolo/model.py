import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class yolov3:
    def _init_(self, img_size, class_num, anchors, use_label_smooth=False):
        self.img_size = img_size
        self.class_num = class_num
        self.anchors = anchors
        self.use_label_smooth = use_label_smooth

    def reorg_layer(self, feature_map, anchors):
        grid_size = tf.shape(feature_map)[1:3]
        ratio = tf.cast(self.img_size // grid_size, tf.float32)
        feature_map = tf.reshape(
            feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self.class_num], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.float32)
        grid_y = tf.range(grid_size[0], dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, 3]), [1, -1, 2])

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * ratio[::-1]

        box_sizes = tf.exp(box_sizes) * anchors

        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (
            feature_map, anchors) in feature_map_anchors]
        x_y_offset, box_predictions, conf_logits, prob_logits = zip(
            *reorg_results)

        x_y_offset = tf.concat(x_y_offset, axis=1)
        box_predictions = tf.concat(box_predictions, axis=1)
        conf_logits = tf.concat(conf_logits, axis=1)
        prob_logits = tf.concat(prob_logits, axis=1)

        return x_y_offset, box_predictions, conf_logits, prob_logits

    def loss_layer(self, feature_map_i, y_true, anchors):
        grid_size = tf.shape(feature_map_i)[1:3]
        ratio = tf.cast(self.img_size // grid_size, tf.float32)
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(
            feature_map_i, anchors)

        pred_boxes = pred_boxes / ratio[::-1]

        iou_scores = self.broadcast_iou(pred_boxes, y_true)
        object_mask = tf.reduce_max(iou_scores, axis=-1)
        object_mask = tf.cast(object_mask >= 0.5, tf.float32)
        ignore_mask = tf.cast((1 - object_mask), tf.float32)

        if self.use_label_smooth:
            positive_mask = object_mask
            negative_mask = ignore_mask
        else:
            positive_mask = object_mask
            negative_mask = ignore_mask

        coord_mask = tf.expand_dims(object_mask, axis=-1) * 5.0
        conf_pos_mask = object_mask * 1.0
        conf_neg_mask = ignore_mask * 1.0
        conf_mask = conf_pos_mask + conf_neg_mask
        prob_mask = object_mask * 1.0

        gt_boxes = y_true[..., :4]
        gt_boxes = gt_boxes * ratio[::-1]

        loss_xy = coord_mask * \
            tf.square(gt_boxes[..., :2] - pred_boxes[..., :2])
        loss_wh = coord_mask * \
            tf.square(tf.sqrt(gt_boxes[..., 2:4]) -
                      tf.sqrt(pred_boxes[..., 2:4]))
        loss_conf = conf_mask * tf.square(object_mask - pred_conf_logits)
        loss_prob = prob_mask * tf.square(y_true[..., 6:] - pred_prob_logits)

        return tf.concat([loss_xy, loss_wh, loss_conf, loss_prob], axis=-1)

    def compute_loss(self, y_pred, y_true):
        self.coord_scale = 1
        self.conf_scale = 1
        self.prob_scale = 1
        loss = 0

        for i in range(3):
            feature_map_i = y_pred[i]
            anchors = self.anchors[(2 - i) * 3:(2 - i) * 3 + 3]
            loss += tf.reduce_mean(self.loss_layer(feature_map_i,
                                   y_true[i], anchors))

        return loss

    def broadcast_iou(self, boxes_1, boxes_2):
        boxes_1 = tf.expand_dims(boxes_1, -2)
        boxes_1_xy = boxes_1[..., :2]
        boxes_1_wh = boxes_1[..., 2:4]
        boxes_1_wh_half = boxes_1_wh / 2.
        boxes_1_mins = boxes_1_xy - boxes_1_wh_half
        boxes_1_maxes = boxes_1_xy + boxes_1_wh_half

        boxes_2 = tf.expand_dims(boxes_2, 0)
        boxes_2_xy = boxes_2[..., :2]
        boxes_2_wh = boxes_2[..., 2:4]
        boxes_2_wh_half = boxes_2_wh / 2.
        boxes_2_mins = boxes_2_xy - boxes_2_wh_half
        boxes_2_maxes = boxes_2_xy + boxes_2_wh_half

        intersect_mins = tf.maximum(boxes_1_mins, boxes_2_mins)
        intersect_maxes = tf.minimum(boxes_1_maxes, boxes_2_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        boxes_1_area = boxes_1_wh[..., 0] * boxes_1_wh[..., 1]
        boxes_2_area = boxes_2_wh[..., 0] * boxes_2_wh[..., 1]
        iou_scores = intersect_area / \
            (boxes_1_area + boxes_2_area - intersect_area)

        return iou_scores
