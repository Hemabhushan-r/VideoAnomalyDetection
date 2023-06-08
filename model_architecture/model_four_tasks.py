#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import tensorflow as tf
import args
import pdb


def decoder_middle_frame(inputs_, is_training):
    upsample1 = tf.image.resize_images(inputs_, size=(inputs_.shape[1] * 2, inputs_.shape[2] * 2),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 32x32x32
    conv_5 = tf.layers.conv2d(upsample1, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_5 = tf.nn.relu(tf.layers.batch_normalization(conv_5, training=is_training))

    upsample2 = tf.image.resize_images(conv_5, size=(conv_5.shape[1] * 2, conv_5.shape[2] * 2),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv_6 = tf.layers.conv2d(upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_6 = tf.nn.relu(tf.layers.batch_normalization(conv_6, training=is_training))

    upsample3 = tf.image.resize_images(conv_6, size=(conv_6.shape[1] * 2, conv_6.shape[2] * 2),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv_7 = tf.layers.conv2d(upsample3, filters=3, kernel_size=(3, 3), padding='same', activation=None)
    conv_7 = tf.nn.relu(tf.layers.batch_normalization(conv_7, training=is_training))
    return conv_7


def decoder_middle_frame_deep(inputs_, is_training):
    upsample1 = tf.image.resize_images(inputs_, size=(inputs_.shape[1] * 2, inputs_.shape[2] * 2),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 32x32x32
    conv_5 = tf.layers.conv2d(upsample1, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_5 = tf.nn.relu(tf.layers.batch_normalization(conv_5, training=is_training))

    upsample2 = tf.image.resize_images(conv_5, size=(conv_5.shape[1] * 2, conv_5.shape[2] * 2),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv_6 = tf.layers.conv2d(upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_6 = tf.nn.relu(tf.layers.batch_normalization(conv_6, training=is_training))

    upsample3 = tf.image.resize_images(conv_6, size=(conv_6.shape[1] * 2, conv_6.shape[2] * 2),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv_8 = tf.layers.conv2d(upsample3, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_8 = tf.nn.relu(tf.layers.batch_normalization(conv_8, training=is_training))

    upsample4 = tf.image.resize_images(conv_8, size=(conv_8.shape[1] * 2, conv_8.shape[2] * 2),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv_7 = tf.layers.conv2d(upsample4, filters=3, kernel_size=(3, 3), padding='same', activation=None)
    conv_7 = tf.nn.relu(tf.layers.batch_normalization(conv_7, training=is_training))
    return conv_7


def decoder_fwd_bwd(inputs_, is_training):
    # 8x8x32
    conv_1 = tf.layers.conv2d(inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
    conv_1 = tf.layers.dropout(conv_1, 0.3, training=is_training)
    
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=[2, 2])

    flat = tf.layers.flatten(max_pool_1)
    logits = tf.layers.dense(flat, units=2, activation=None)

    return logits


def decoder_consecutive(inputs_, is_training):
    # 8x8x32
    conv_1 = tf.layers.conv2d(inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
    conv_1 = tf.layers.dropout(conv_1, 0.3, training=is_training)

    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=[2, 2])

    flat = tf.layers.flatten(max_pool_1)
    logits = tf.layers.dense(flat, units=2, activation=None)

    return logits


def decoder_resnet(inputs_, is_training):
    # 8x8x32
    conv_1 = tf.layers.conv2d(inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
    conv_1 = tf.layers.dropout(conv_1, 0.3, training=is_training)

    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=[2, 2])

    flat = tf.layers.flatten(max_pool_1)
    logits = tf.layers.dense(flat, units=1080, activation=tf.nn.relu)

    return logits


def decoder_c3d(inputs_, is_training):
    # 8x8x32
    conv_1 = tf.layers.conv2d(inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=None)
    conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
    conv_1 = tf.layers.dropout(conv_1, 0.3, training=is_training)

    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=[2, 2])

    flat = tf.layers.flatten(max_pool_1)
    logits = tf.layers.dense(flat, units=487, activation=tf.nn.relu)

    return logits


def encoder(inputs_, is_training, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        # kernel_size(depth, height and width)
        conv_1 = tf.layers.conv3d(inputs_, filters=16, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
        # now shape batch, 5, 64, 64, 16
        # conv_1 = tf.layers.dropout(conv_1, 0.2, training=is_training)

        # pool_size(depth, height and width)
        max_pool_1 = tf.layers.max_pooling3d(conv_1, pool_size=[1, 2, 2], strides=[1, 2, 2])
        # now shape batch, 5, 32, 32, 16

        conv_2 = tf.layers.conv3d(max_pool_1, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=is_training))
        # conv_2 = tf.layers.dropout(conv_2, 0.3, training=is_training)
        # now shape batch, 3, 32, 32, 32

        max_pool_2 = tf.layers.max_pooling3d(conv_2, pool_size=[1, 2, 2], strides=[1, 2, 2])
        # now shape batch, 3, 16, 16, 32

        conv_3 = tf.layers.conv3d(max_pool_2, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=is_training))
        # conv_3 = tf.layers.dropout(conv_3, 0.3, training=is_training)
        # now shape batch, 1, 16, 16, 32

        temporal_max_pooling = tf.layers.max_pooling3d(conv_3, pool_size=[conv_3.shape[1], 2,  2], strides=2)
        # pdb.set_trace()
        encoded = tf.reshape(temporal_max_pooling, (-1, 8, 8, 32))
        return encoded


def encoder_deep(inputs_, is_training, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        # kernel_size(depth, height and width)
        conv_1 = tf.layers.conv3d(inputs_, filters=16, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
        conv_1_1 = tf.layers.conv3d(conv_1, filters=16, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_1_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1_1, training=is_training))
        # now shape batch, 5, 64, 64, 16
        # conv_1 = tf.layers.dropout(conv_1, 0.2, training=is_training)

        # pool_size(depth, height and width)
        max_pool_1 = tf.layers.max_pooling3d(conv_1_1, pool_size=[1, 2, 2], strides=[1, 2, 2])
        # now shape batch, 5, 32, 32, 16

        conv_2 = tf.layers.conv3d(max_pool_1, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=is_training))
        conv_2_1 = tf.layers.conv3d(conv_2, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_2_1 = tf.nn.relu(tf.layers.batch_normalization(conv_2_1, training=is_training))

        # now shape batch, 3, 32, 32, 32
        max_pool_2 = tf.layers.max_pooling3d(conv_2_1, pool_size=[1, 2, 2], strides=[1, 2, 2])
        # now shape batch, 3, 16, 16, 32

        conv_3 = tf.layers.conv3d(max_pool_2, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=is_training))

        max_pool_3 = tf.layers.max_pooling3d(conv_3, pool_size=[1, 2, 2], strides=[1, 2, 2])

        conv_4 = tf.layers.conv3d(max_pool_3, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_4 = tf.nn.relu(tf.layers.batch_normalization(conv_4, training=is_training))

        temporal_max_pooling = tf.layers.max_pooling3d(conv_4, pool_size=[conv_4.shape[1], 2,  2], strides=2)

        encoded = tf.reshape(temporal_max_pooling, (-1, 4, 4, 32))
        return encoded


def encoder_deep_wide(inputs_, is_training, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        # kernel_size(depth, height and width)
        conv_1 = tf.layers.conv3d(inputs_, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
        conv_1_1 = tf.layers.conv3d(conv_1, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_1_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1_1, training=is_training))
        # now shape batch, 5, 64, 64, 16
        # conv_1 = tf.layers.dropout(conv_1, 0.2, training=is_training)

        # pool_size(depth, height and width)
        max_pool_1 = tf.layers.max_pooling3d(conv_1_1, pool_size=[1, 2, 2], strides=[1, 2, 2])
        # now shape batch, 5, 32, 32, 16

        conv_2 = tf.layers.conv3d(max_pool_1, filters=64, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=is_training))
        conv_2_1 = tf.layers.conv3d(conv_2, filters=64, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_2_1 = tf.nn.relu(tf.layers.batch_normalization(conv_2_1, training=is_training))
        # conv_2 = tf.layers.dropout(conv_2, 0.3, training=is_training)
        # now shape batch, 3, 32, 32, 32

        max_pool_2 = tf.layers.max_pooling3d(conv_2_1, pool_size=[1, 2, 2], strides=[1, 2, 2])
        # now shape batch, 3, 16, 16, 32

        conv_3 = tf.layers.conv3d(max_pool_2, filters=64, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=is_training))

        max_pool_3 = tf.layers.max_pooling3d(conv_3, pool_size=[1, 2, 2], strides=[1, 2, 2])

        conv_4 = tf.layers.conv3d(max_pool_3, filters=64, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_4 = tf.nn.relu(tf.layers.batch_normalization(conv_4, training=is_training))

        temporal_max_pooling = tf.layers.max_pooling3d(conv_4, pool_size=[conv_4.shape[1], 2,  2], strides=2)
        # pdb.set_trace()
        encoded = tf.reshape(temporal_max_pooling, (-1, 4, 4, 64))
        return encoded


def encoder_wide(inputs_, is_training, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        # kernel_size(depth, height and width)
        conv_1 = tf.layers.conv3d(inputs_, filters=32, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))
        # now shape batch, 5, 64, 64, 16
        

        # pool_size(depth, height and width)
        max_pool_1 = tf.layers.max_pooling3d(conv_1, pool_size=[1, 2, 2], strides=[1, 2, 2])
        # now shape batch, 5, 32, 32, 16

        conv_2 = tf.layers.conv3d(max_pool_1, filters=64, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=is_training))

        # now shape batch, 3, 32, 32, 32
        max_pool_2 = tf.layers.max_pooling3d(conv_2, pool_size=[1, 2, 2], strides=[1, 2, 2])

        # now shape batch, 3, 16, 16, 32
        conv_3 = tf.layers.conv3d(max_pool_2, filters=64, kernel_size=(3, 3, 3), padding='same', activation=None)
        conv_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=is_training))

        # now shape batch, 1, 16, 16, 32
        temporal_max_pooling = tf.layers.max_pooling3d(conv_3, pool_size=[conv_3.shape[1], 2,  2], strides=2)
        # pdb.set_trace()
        encoded = tf.reshape(temporal_max_pooling, (-1, 8, 8, 64))
        return encoded


def model(inputs_, inputs_consecutive, input_resnet, is_training, use_temp=False):
    encoded = encoder(inputs_, is_training)
    decoder_middle_ = decoder_middle_frame(encoded, is_training)
    logits_fwd_bwd = decoder_fwd_bwd(encoded, is_training)
    
    encoded_consecutive = encoder(inputs_consecutive, is_training, True)
    logits_consecutive = decoder_consecutive(encoded_consecutive, is_training)

    encoded_resnet = encoder(input_resnet, is_training, True)
    logits_resnet = decoder_resnet(encoded_resnet, is_training)

    if use_temp:
        print('!!!!!!!!!!!! temp used')
        logits_resnet = tf.math.divide(logits_resnet, args.temperature)

    return decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet


def model_wide(inputs_, inputs_consecutive, input_resnet, is_training):
    encoded = encoder_wide(inputs_, is_training)
    decoder_middle_ = decoder_middle_frame(encoded, is_training)
    logits_fwd_bwd = decoder_fwd_bwd(encoded, is_training)

    encoded_consecutive = encoder_wide(inputs_consecutive, is_training, True)
    logits_consecutive = decoder_consecutive(encoded_consecutive, is_training)

    encoded_resnet = encoder_wide(input_resnet, is_training, True)
    logits_resnet = decoder_resnet(encoded_resnet, is_training)

    return decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet


def model_deep(inputs_, inputs_consecutive, input_resnet, is_training):
    encoded = encoder_deep(inputs_, is_training)
    decoder_middle_ = decoder_middle_frame_deep(encoded, is_training)
    logits_fwd_bwd = decoder_fwd_bwd(encoded, is_training)

    encoded_consecutive = encoder_deep(inputs_consecutive, is_training, True)
    logits_consecutive = decoder_consecutive(encoded_consecutive, is_training)

    encoded_resnet = encoder_deep(input_resnet, is_training, True)
    logits_resnet = decoder_resnet(encoded_resnet, is_training)
    # pdb.set_trace()
    return decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet


def model_deep_wide(inputs_, inputs_consecutive, input_resnet, is_training):
    encoded = encoder_deep_wide(inputs_, is_training)
    decoder_middle_ = decoder_middle_frame_deep(encoded, is_training)
    logits_fwd_bwd = decoder_fwd_bwd(encoded, is_training)

    encoded_consecutive = encoder_deep_wide(inputs_consecutive, is_training, True)
    logits_consecutive = decoder_consecutive(encoded_consecutive, is_training)

    encoded_resnet = encoder_deep_wide(input_resnet, is_training, True)
    logits_resnet = decoder_resnet(encoded_resnet, is_training)

    return decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet


def model_deep_wide_c3d(inputs_, inputs_consecutive, input_resnet, inputs_c3d, is_training):
    encoded = encoder_deep_wide(inputs_, is_training)
    decoder_middle_ = decoder_middle_frame_deep(encoded, is_training)
    logits_fwd_bwd = decoder_fwd_bwd(encoded, is_training)

    encoded_consecutive = encoder_deep_wide(inputs_consecutive, is_training, True)
    logits_consecutive = decoder_consecutive(encoded_consecutive, is_training)

    encoded_resnet = encoder_deep_wide(input_resnet, is_training, True)
    logits_resnet = decoder_resnet(encoded_resnet, is_training)

    encoded_c3d = encoder_deep_wide(inputs_c3d, is_training, True)
    logits_c3d = decoder_c3d(encoded_c3d, is_training)

    return decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet, logits_c3d




