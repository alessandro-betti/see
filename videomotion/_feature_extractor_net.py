import tensorflow as tf
import numpy as np
import os
import shutil
# python vprocessor.py --run ../data/skater.avi --out exp/skater1 --gray 1 --save_scores_only 0 --res 240x180 --rep 1000000 --all_black 0 --m 10 --f 7 --init_q 1.0 --k 0.000001 --lambdaE 2.0 --lambdaC 1.0 --step_size 0.1 --port 8888


class FeatureExtractor:

    def __init__(self, w, h, options, resume=False):
        self.step = 1

        # saving options (and other useful values) to direct attributes
        self.w = w
        self.h = h
        self.wh = w*h  # number of pixels (per input channel)
        self.step_size = options['step_size']
        self.step_adapt = options['step_adapt']
        self.f = options['f']  # full edge of the filter (i.e., 3 in case of 3x3 filters)
        self.n = options['n']  # input channels/features
        self.m = options['m']  # output features
        self.ffn = self.f * self.f * self.n  # unrolled filter volume
        self.init_q = options['init_q']
        self.k = options['k']
        self.lambdaE = options['lambdaE']
        self.lambdaC = options['lambdaC']
        self.gew = options['gew']
        self.all_black = options['all_black']
        self.init_fixed = options['init_fixed'] > 0
        self.resume = resume

        # TensorFlow session and graph
        self.sess = tf.Session()
        self.process_frame_ops, self.saver = self.__build_graph()

        # TensorBoard-related
        if os.path.exists(options['root'] + '/tensor_board'):
            shutil.rmtree(options['root'] + '/tensor_board')
        self.summary_writer = tf.summary.FileWriter(options['root'] + '/tensor_board', self.sess.graph)

        # TensorFlow model, save dir
        self.save_path = options['root'] + '/model/model.saved'

    def close(self):
        self.sess.close()

    def save(self):
        self.saver.save(self.sess, self.save_path)

    def load(self, steps):
        self.saver.restore(self.sess, self.save_path)
        self.step = steps

    # processing the current frame (frame_1 below)
    def run_step(self, frame_0_to_feed, frame_1_to_feed, motion_01_to_feed):

        # zero signal
        if self.all_black > 0:
            frame_0_to_feed.fill(0.0)
            frame_1_to_feed.fill(0.0)
            motion_01_to_feed.fill(0.0)

        # getting values that are fed to the session "runner"
        fed_frame = tf.get_default_graph().get_tensor_by_name("fed_frame:0")
        feed_dict = {fed_frame: frame_1_to_feed}

        # running the computations over the TensorFlow graph
        feature_maps, filters_matrix, \
            mi, ce, minus_ge, norm_q, all_terms, _, \
            summary_ops = self.sess.run(self.process_frame_ops, feed_dict=feed_dict)

        # TensorBoard-related
        self.summary_writer.add_summary(summary_ops, self.step)

        # next step
        self.step = self.step + 1

        # returning data (no output printing in this class, please!)
        return feature_maps, filters_matrix, \
            mi, mi, ce, minus_ge, 0.0, 0.0, 0.0, norm_q, 0.0, 0.0, 0.0, 0.0, \
            all_terms, 0.0, 1.0

    def __build_graph(self):

        # precision
        precision = tf.float32

        # inputs
        fed_frame = tf.placeholder(precision, shape=[self.h, self.w, self.n], name="fed_frame")

        # TensorFlow variables (main scope)
        with tf.variable_scope("main", reuse=False) as my_scope:

            # prepare the input data
            frame = tf.expand_dims(tf.div(fed_frame, 255.0), 0)

            # convolutional layer
            use_bias = False
            if not self.init_fixed:
                initializer = tf.random_uniform_initializer(-self.init_q, self.init_q)
            else:
                initializer = tf.constant_initializer(self.init_q)

            feature_acts = tf.layers.conv2d(inputs=frame,
                                            filters=self.m, kernel_size=self.f, strides=(1,1), padding='SAME',
                                            data_format='channels_last',
                                            activation=None, kernel_initializer=initializer,
                                            use_bias=use_bias, bias_initializer=tf.zeros_initializer(),
                                            trainable=True,
                                            name='conv1')

            # reshaping
            logits = tf.reshape(feature_acts, [self.wh, self.m])

            # softmax
            feature_maps = tf.nn.softmax(logits, dim=1)

            # getting internal data
            filters_matrix = tf.get_default_graph().get_tensor_by_name('main/conv1/kernel:0')
            if use_bias:
                filters_bias = tf.get_default_graph().get_tensor_by_name('main/conv1/bias:0')
            else:
                filters_bias = tf.zeros(shape=[self.f])

            # objective function terms: ce, -ge, mi
            p = tf.maximum(feature_maps, 0.00001)
            p_log_p = tf.multiply(p, tf.div(tf.log(p), np.log(self.m)))  # wh x m
            avg_p = tf.reduce_mean(p, 0)  # m
            ce = -tf.reduce_sum(tf.reduce_mean(p_log_p, 0))
            minus_ge = tf.reduce_sum(tf.multiply(avg_p, tf.div(tf.log(avg_p), np.log(self.m))))
            norm_q = tf.reduce_sum(tf.square(filters_matrix)) + tf.reduce_sum(tf.square(filters_bias))
            mi = -ce-minus_ge

            # objective function
            obj = self.lambdaC * 0.5 * ce + self.lambdaE * 0.5 * minus_ge + self.k * norm_q

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size)
            train_op = optimizer.minimize(obj)

            # TensorBoard-related
            tf.summary.scalar('A_MutualInformation', mi)
            tf.summary.scalar('C_ConditionalEntropy', ce)
            tf.summary.scalar('D_MinusEntropy', minus_ge)
            tf.summary.scalar('E_NormQ', norm_q)
            tf.summary.scalar("B_FullObjectiveFunction", obj)

            # operations to be executed in the data flow graph (filters_matrix: filter_volume x m)
            out_feature_maps = tf.reshape(feature_maps, [self.h, self.w, self.m])  # h x w x m
            out_filters_map = tf.transpose(tf.reshape(filters_matrix, [self.f * self.f, self.n, self.m]))  # m x n x f^2

            # summaries
            summary_ops = tf.summary.merge_all()

            ops = [out_feature_maps,
                   out_filters_map,
                   mi, ce, minus_ge, norm_q,
                   obj, train_op,
                   summary_ops]

            # initialization
            if not self.resume:
                self.sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

        return ops, saver

