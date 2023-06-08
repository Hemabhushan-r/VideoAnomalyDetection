#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import glob
import os
import tempfile

import cv2 as cv
import numpy as np
from tensorflow.keras import activations
from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import Model, load_model

import args
import utils


def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)


def remove_the_softmax_function(model):
    new_layer = model.layers[-1]
    new_layer.activation = activations.linear
    new_model = Model(model.input, new_layer.output)
    new_model = apply_modifications(new_model)
    return new_model


def extract():
    # create network
    model = resnet50.ResNet50()
    model = remove_the_softmax_function(model)

    folder_base = os.path.join(args.output_folder_base, args.database_name, utils.ProcessingType.TRAIN.value)
    output_folder_base = os.path.join(args.output_folder_base, args.database_name, utils.ProcessingType.TRAIN.value,
                                      '%s', args.imagenet_logits_folder_name)

    videos_names = os.listdir(folder_base)
    for video_name in videos_names:
        video_samples = glob.glob(os.path.join(folder_base, video_name, args.samples_folder_name, '*.npy'))
        output_folder = output_folder_base % video_name
        utils.create_dir(output_folder)
        utils.log_message(video_name)
        for video_sample in video_samples:
            if video_sample.find('_64.npy') != -1:  # skip repeated files
                continue

            short_file_name = video_sample.split(os.sep)[-1]
            sample = np.load(video_sample)

            img = sample[15]
            img = cv.resize(img, (224, 224))
            x = np.expand_dims(img, axis=0)
            # Scale the input image to the range used in the trained network
            x = resnet50.preprocess_input(x)
            # Run the image through the deep neural network to make a prediction
            predictions = model.predict(x)
            np.save(os.path.join(output_folder, short_file_name), predictions)



