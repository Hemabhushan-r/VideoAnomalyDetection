#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import os
import numpy as np
import sklearn.cluster as sc
from sklearn.metrics import roc_curve, auc
import timeit
import pickle
import pdb
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve
import math
import cv2 as cv
import random
import matplotlib.pyplot as plt

from utils import log_function_start, log_function_end, log_message, create_dir, ProcessingType, train_linear_svm,\
    check_file_existence
import args


def normalize_(err_):
    err_ = np.array(err_)
    err_ = err_ - min(err_)
    err_ = err_ / max(err_)
    return err_


def compute_anomaly_scores_per_object(processing_type: ProcessingType, save_per_video=True):
    log_function_start() 

    meta_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                 args.meta_folder_name, "%s")
    video_patch_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value,
                                        "%s", args.samples_folder_name, '%s')  # args.samples_folder_name

    videos_features_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value)

    videos_names = os.listdir(videos_features_base_dir)
    videos_names.sort()
    concat_features_path = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                        "anormality_scores.txt")
    loc_v3_path = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                               "loc_v3.npy")

    import middle_fbwd_consecutive_resnet.trainer as trainer
    discriminator = trainer.Experiment(is_testing=True)

    all_features = []
    for video_name in videos_names:  # for each video
        # check if it is a dir
        if os.path.isdir(os.path.join(videos_features_base_dir, video_name)) is False:
            continue
        log_message(video_name)
        # read all the appearance features
        samples_names = os.listdir(meta_base_dir % (video_name, ""))  # maybe this is a hack :)
        samples_names.sort()
        if save_per_video:
            video_features_path = concat_features_path % video_name
            video_loc_v3_path = loc_v3_path % video_name

        features_video = []
        errs_1 = []
        errs_2 = []
        errs_3 = []
        errs_4 = []
        loc_v3_video = []
        for sample_name in samples_names:
            try:
                video_patch = np.load(video_patch_base_dir % (video_name, sample_name.replace('.txt', '.npy')))
                meta = np.loadtxt(meta_base_dir % (video_name, sample_name))
            except:
                log_message(sample_name)
                continue

            err_1, err_2, err_3, err_4 = discriminator.get_normality_score(video_patch, meta)
            errs_1.append(err_1)
            errs_2.append(err_2)
            errs_3.append(err_3)
            errs_4.append(err_4)

            loc_v3_video.append(meta[:-2])

        errs_1 = normalize_(errs_1)
        errs_2 = normalize_(errs_2)
        errs_3 = normalize_(errs_3)
        errs_4 = normalize_(errs_4)
        
        features_video = np.array(errs_1) + np.array(errs_2) + np.array(errs_3) + np.array(errs_4)
        if save_per_video:
            np.savetxt(video_features_path, features_video)
            np.save(video_loc_v3_path, loc_v3_video)

    # save the features
    if not save_per_video:
        np.save(concat_features_path, all_features)
    log_function_end()


def gaussian_filter_3d(sigma=1.0):
    x = np.array([-2, -1, 0, 1, 2])
    f = np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
    f += (1 - np.sum(f)) / len(f)
    k = np.expand_dims(f, axis=1).T * np.expand_dims(f, axis=1)
    k3d = np.expand_dims(k, axis=2).T * np.expand_dims(np.expand_dims(f, axis=1), axis=2)
    # k3d = k3d * 3
    return k3d


def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter


def predict_anomaly_on_frames(video_info_path, filter_3d, filter_2d):
    video_normality_scores = np.loadtxt(os.path.join(video_info_path, "anormality_scores.txt"))
    video_loc_v3 = np.load(os.path.join(video_info_path, "loc_v3.npy"))
    video_meta_data = pickle.load(open(os.path.join(video_info_path, "video_meta_data.pkl"), 'rb'))
    video_height = video_meta_data["height"]
    video_width = video_meta_data["width"]

    block_scale = args.block_scale
    block_h = int(round(video_height / block_scale))
    block_w = int(round(video_width / block_scale))

    anomaly_scores = video_normality_scores - min(video_normality_scores)
    anomaly_scores = anomaly_scores / max(anomaly_scores)

    num_frames = video_meta_data["num_frames"]
    num_bboxes = len(anomaly_scores)

    ab_event = np.zeros((block_h, block_w, num_frames))
    for i in range(num_bboxes):
        loc_V3 = np.int32(video_loc_v3[i])

        ab_event[int(round(loc_V3[2] / block_scale)): int(round(loc_V3[4] / block_scale)) + 1,
        int(round(loc_V3[1] / block_scale)): int(round(loc_V3[3] / block_scale) + 1), loc_V3[0]] = np.maximum(
            ab_event[int(round(loc_V3[2] / block_scale)):int(round(loc_V3[4] / block_scale)) + 1,
            int(round(loc_V3[1] / block_scale)): int(round(loc_V3[3] / block_scale)) + 1,
            loc_V3[0]], anomaly_scores[i])
    

    dim = 9
    filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)
    ab_event3 = convolve(ab_event, filter_3d)  #  ab_event.copy() #
    np.save(os.path.join(video_info_path, 'ab_event3.npy'), ab_event3)
    frame_scores = np.zeros(num_frames)
    for i in range(num_frames):
        frame_scores[i] = ab_event3[:, :, i].max()

    padding_size = len(filter_2d) // 2
    # np.savetxt('anomaly_on_frames/' + video_info_path.split(os.sep)[-1] + '.txt', frame_scores)
    # in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    in_ = np.concatenate((frame_scores[:padding_size], frame_scores, frame_scores[-padding_size:]))
    frame_scores = np.correlate(in_, filter_2d, 'valid')
    return frame_scores


def compute_performance_indices(processing_type:ProcessingType=ProcessingType.TEST):
    log_function_start()
    filter_3d = gaussian_filter_3d(sigma=25)  # don't use it here
    filter_2d = gaussian_filter_(np.arange(1, 50), 20)

    # list all the testing videos
    videos_features_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value)
    testing_videos_names =[name for name in os.listdir(videos_features_base_dir) if os.path.isdir(os.path.join(videos_features_base_dir, name))]
    testing_videos_names.sort()
    all_frame_scores = []
    all_gt_frame_scores = []
    roc_auc_videos = []
    
    for video_name in testing_videos_names:
        log_message(video_name)
        video_scores = predict_anomaly_on_frames(os.path.join(videos_features_base_dir, video_name), filter_3d, filter_2d)
        all_frame_scores = np.append(all_frame_scores, video_scores)
        # read the ground truth scores at frame level
        gt_scores = np.loadtxt(os.path.join(videos_features_base_dir, video_name, "ground_truth_frame_level.txt"))
        all_gt_frame_scores = np.append(all_gt_frame_scores, gt_scores)
        fpr, tpr, _ = roc_curve(np.concatenate(([0], gt_scores, [1])), np.concatenate(([0], video_scores, [1])))
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        roc_auc_videos.append(roc_auc)
    # plt.plot(all_gt_frame_scores)
    # plt.plot(all_frame_scores)
    # plt.show()

    fpr, tpr, _ = roc_curve(all_gt_frame_scores, all_frame_scores)
    roc_auc = auc(fpr, tpr)
    log_message("Frame-based AUC is %.3f on %s (all data set)." % (roc_auc, args.database_name))
    log_message("Avg. (on video) frame-based AUC is %.3f on %s." % (np.array(roc_auc_videos).mean(), args.database_name))
    log_function_end()

