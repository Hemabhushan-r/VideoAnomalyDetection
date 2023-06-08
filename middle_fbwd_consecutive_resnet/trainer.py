#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import os
import re

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import args
import model_architecture.model_four_tasks as model
import utils
from middle_fbwd_consecutive_resnet.data_set_reader import create_readers_split


class Experiment:

    def __init__(self, learning_rate_init=10 ** -3, num_epochs=30, batch_size=64, is_testing=False, checkpoint_epoch=None):
        tf.reset_default_graph()
        self.input_size_h = 64
        self.input_size_w = 64
        self.input_size_d = int(7)
        self.learning_rate_init = learning_rate_init
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_folder = os.path.join(args.CHECKPOINTS_BASE, "network_lr_%.5f_ep_%d_batch_%d" %
                                              (learning_rate_init, num_epochs, batch_size))

        self.IS_RESTORE = tf.train.latest_checkpoint(self.checkpoint_folder) is not None

        self.inputs = tf.placeholder(np.float32,  (None, 6, self.input_size_h, self.input_size_w, 3))
        self.inputs_consecutive = tf.placeholder(np.float32,  (None, 7, self.input_size_h, self.input_size_w, 3))
        self.inputs_resnet = tf.placeholder(np.float32,  (None, 7, self.input_size_h, self.input_size_w, 3))

        self.is_training = tf.placeholder(np.bool, None)

        self.targets_decoder = tf.placeholder(np.float32, [None, self.input_size_h, self.input_size_w, 3])
        self.targets_fwd_bwd = tf.placeholder(np.float32, [None, 2])
        self.targets_consecutive = tf.placeholder(np.float32, [None, 2])
        self.targets_resnet = tf.placeholder(np.float32, [None, 1080])

        # build neural network
        self.decoder_middle_, self.logits_fbwd, self.logits_consecutive, self.logits_resnet \
            = model.model_deep_wide(self.inputs, self.inputs_consecutive, self.inputs_resnet ,self.is_training)

        self.cost = tf.reduce_mean(tf.abs(self.decoder_middle_ - self.targets_decoder)) \
                    + tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_fbwd, labels=self.targets_fwd_bwd)\
                    + tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_consecutive, labels=self.targets_consecutive) \
                    + 0.5 * tf.reduce_mean(tf.abs(self.logits_resnet - self.targets_resnet))

        self.avg_cost = tf.reduce_mean(self.cost)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate_init
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.avg_cost,
                                                                                               global_step=self.global_step)
        self.sess = tf.Session(config=args.tf_config)

        self.train_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="train_loss")
        self.val_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="val_loss")

        self.train_acc_placeholder = tf.placeholder(tf.float32, shape=[], name="train_acc")
        self.val_acc_placeholder = tf.placeholder(tf.float32, shape=[], name="val_acc")

        tf.summary.scalar('train_loss', self.train_loss_placeholder)
        tf.summary.scalar('val_loss', self.val_loss_placeholder)

        # tf.summary.scalar('train_acc', self.train_acc_placeholder)
        # tf.summary.scalar('val_acc', self.val_acc_placeholder)
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.merged = tf.summary.merge_all()
        if is_testing:
            if checkpoint_epoch is not None:
                self.restore_model(checkpoint_epoch)
            else:
                self.restore_model()
 
    def fit(self, reader_train):
        iters = int(np.ceil(reader_train.num_samples / self.batch_size))
        total_loss = 0
        un_norm_acc_fwd = 0
        un_norm_acc_consecutive = 0
        print(iters)
        for iter in range(iters):
            # utils.log_message(str(iter))
            # print(iter, '\r')
            batch_x, batch_x_consecutive,  batch_x_resnet,  \
            batch_y_decoder, batch_y_fwd_bwd, batch_y_consecutive, batch_y_resnet \
                = reader_train.next_batch(self.batch_size)
            
            _, c, predictions_fwd_bwd, predictions_consecutive, _ = self.sess.run([self.optimizer,
                                                                                   self.cost,
                                                                                   self.logits_fbwd,
                                                                                   self.logits_consecutive,
                                                                                   self.global_step],
                                                 feed_dict={self.inputs: batch_x,
                                                            self.inputs_consecutive: batch_x_consecutive,
                                                            self.inputs_resnet: batch_x_resnet,
                                                            self.targets_decoder: batch_y_decoder,
                                                            self.targets_fwd_bwd: batch_y_fwd_bwd,
                                                            self.targets_consecutive: batch_y_consecutive,
                                                            self.targets_resnet: batch_y_resnet,
                                                            self.is_training: True})
            
            total_loss += np.sum(c)
            un_norm_acc_fwd += np.sum(np.argmax(predictions_fwd_bwd, axis=1) == np.argmax(batch_y_fwd_bwd, axis=1))
            un_norm_acc_consecutive += np.sum(np.argmax(predictions_consecutive, axis=1) == np.argmax(batch_y_consecutive, axis=1))

        return total_loss / reader_train.num_samples, un_norm_acc_fwd / reader_train.num_samples,\
               un_norm_acc_consecutive / reader_train.num_samples

    def get_normality_score(self, video_patch, meta):
        resized_sample = []
        for i in range(len(video_patch)):
            sample_obj = video_patch[i]
            sample_obj = cv.resize(sample_obj, (self.input_size_w, self.input_size_w),
                                   interpolation=cv.INTER_CUBIC)
            resized_sample.append(sample_obj)
        full_sample = np.array(resized_sample) / 255.0
        
        label = full_sample[args.temporal_size]
        sample = np.delete(full_sample, args.temporal_size, 0)
        logits_fwd, reconstruction, logits_cons, logits_resnet = self.sess.run([self.logits_fbwd, self.decoder_middle_, self.logits_consecutive, self.logits_resnet],
                                                   feed_dict={self.inputs: [sample],
                                                              self.inputs_consecutive: [full_sample],
                                                              self.inputs_resnet: [full_sample],
                                                              self.is_training: False})
        reconstruction = reconstruction[0]
        probs_fwd = utils.softmax(logits_fwd[0])
        probs_con = utils.softmax(logits_cons[0])
        class_id = int(meta[-2]) - 1

        return np.mean(np.abs(label - reconstruction)), probs_fwd[1],  probs_con[1], np.abs(meta[-1] - logits_resnet[0, 1000 + class_id])

    def eval(self, reader, return_predicted_labels=False):
        iters = int(np.ceil(reader.num_samples / self.batch_size))
        total_loss = 0
        un_norm_acc_fwd = 0
        un_norm_acc_consecutive = 0
        loss_resnet = 0
        loss_reconstruction = 0

        for iter in range(iters):
            batch_x, batch_x_consecutive, batch_x_resnet,  batch_y_decoder, batch_y_fwd_bwd, batch_y_consecutive, batch_y_resnet \
                = reader.next_batch(self.batch_size)

            c, predictions_fwd_bwd, predictions_consecutive, decoder_middle_, predictions_resnet_\
                = self.sess.run([self.cost, self.logits_fbwd,  self.logits_consecutive, self.decoder_middle_, self.logits_resnet],
                                  feed_dict={self.inputs: batch_x,
                                             self.inputs_consecutive: batch_x_consecutive,
                                             self.targets_decoder: batch_y_decoder,
                                             self.targets_fwd_bwd: batch_y_fwd_bwd,
                                             self.targets_consecutive: batch_y_consecutive,
                                             self.inputs_resnet: batch_x_resnet,
                                             self.targets_resnet: batch_y_resnet,
                                             self.is_training: False})

            # pdb.set_trace()
            loss_resnet += np.sum(np.mean(np.abs(batch_y_resnet - predictions_resnet_), axis=1))
            loss_reconstruction += np.sum(np.mean(np.abs(batch_y_decoder - decoder_middle_), axis=(1, 2, 3)))
            total_loss += np.sum(c)
            un_norm_acc_fwd += np.sum(np.argmax(predictions_fwd_bwd, axis=1) == np.argmax(batch_y_fwd_bwd, axis=1))
            un_norm_acc_consecutive += np.sum(
                np.argmax(predictions_consecutive, axis=1) == np.argmax(batch_y_consecutive, axis=1))

        return total_loss / reader.num_samples, un_norm_acc_fwd / reader.num_samples, \
               un_norm_acc_consecutive / reader.num_samples, loss_resnet / reader.num_samples,\
               loss_reconstruction / reader.num_samples

    def restore_model(self, epoch=None):
        if epoch is None:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_folder)
        else:
            checkpoint_path = os.path.join(self.checkpoint_folder, "model_%d" % epoch)
        
        if checkpoint_path is None:
            raise Exception("Checkpoint file is missing!")

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(self.sess, checkpoint_path)

    def get_statistics_set(self, reader, epoch=None):
        self.restore_model(epoch=epoch)
        loss, acc, predicted_labels = self.eval(reader, return_predicted_labels=True)
        utils.log_message('loss = {}, acc = {} \nconf mat = \n{}'.format(loss, acc, confusion_matrix(np.argmax(reader.labels, axis=1), predicted_labels)))

    # def get_statistics(self, epoch=None):
    #     utils.log_message("Statistics for epoch: {}".format(epoch))
    #     utils.log_message("TRAINING")
    #     self.get_statistics_set(reader_train, epoch)
    #     utils.log_message("VAL")
    #     self.get_statistics_set(reader_val, epoch)

    def run(self, reader_train, reader_val):
        start_epoch = 0
        saver = tf.train.Saver(max_to_keep=0)
        self.sess.run(tf.global_variables_initializer())
        if self.IS_RESTORE:
            print('=' * 30 + '\nRestoring from ' + tf.train.latest_checkpoint(self.checkpoint_folder))
            saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = int(start_epoch[-1]) + 1

        writer = tf.summary.FileWriter(os.path.join(args.CHECKPOINTS_BASE, 'train_disc.log'), self.sess.graph)

        for epoch in range(start_epoch, self.num_epochs):
            utils.log_message("Epoch: %d/%d" % (epoch, self.num_epochs))
            train_loss, train_acc_1, train_acc_2 = self.fit(reader_train)
            val_loss, val_acc_1,  val_acc_2, loss_resnet, loss_recon = self.eval(reader_val)
            utils.log_message("loss train = %.4f, val = %.4f" % (train_loss, val_loss))
            utils.log_message("acc fwd train = %.4f, val = %.4f" % (train_acc_1, val_acc_1))
            utils.log_message("acc cons train = %.4f, val = %.4f" % (train_acc_2, val_acc_2))
            utils.log_message("loss_resnet = %.4f, loss_recon = %.4f" % (loss_resnet, loss_recon))
            merged_ = self.sess.run(self.merged, feed_dict={
                                                            self.train_loss_placeholder: train_loss,
                                                            self.val_loss_placeholder: val_loss })
            writer.add_summary(merged_, epoch)

            saver.save(self.sess, os.path.join(self.checkpoint_folder, "model_%d" % epoch))


def train():
    exp = Experiment()
    reader_train, reader_val = create_readers_split()
    exp.run(reader_train, reader_val)


def eval(epoch):
    exp = Experiment()
    reader_train, reader_val = create_readers_split()
    exp.restore_model(epoch)
    total_loss, acc_fwd, acc_cons, loss_resnet, loss_recon = exp.eval(reader_val)
    utils.log_message("total_loss eval = %.4f" % total_loss)
    utils.log_message("acc_fwd eval = %.4f" % acc_fwd)
    utils.log_message("acc_cons eval = %.4f" % acc_cons)
    utils.log_message("loss_resnet eval = %.4f" % loss_resnet)
    utils.log_message("loss_recon eval = %.4f" % loss_recon)
