import tensorflow as tf
import math
import numpy as np
import os
import shutil
import json


class FeatureExtractor:

    def __init__(self, w, h, options, resume=False):
        self.__first_frame = True
        self.w = w
        self.h = h
        self.step = 1
        self.wh = w*h  # number of pixels (per input channel)
        self.step_size = options['step_size']
        self.step_adapt = options['step_adapt']
        self.f = options['f']  # full edge of the filter (i.e., 3 in case of 3x3 filters)
        self.n = options['n']  # input channels/features
        self.m = options['m']  # output features
        self.init_q = options['init_q']
        self.alpha = options['alpha']
        self.beta = options['beta']
        self.theta = options['theta']
        self.k = options['k']
        self.gamma = options['gamma']
        self.lambdaE = options['lambdaE']
        self.lambdaC = options['lambdaC']
        self.lambdaM = options['lambdaM']
        self.lambda0 = options['lambda0']
        self.lambda1 = options['lambda1']
        self.eps1 = options['eps1']
        self.eps2 = options['eps2']
        self.eps3 = options['eps3']
        self.zeta = options['zeta']
        self.eta = options['eta']
        self.all_black = options['all_black']
        self.init_fixed = options['init_fixed'] > 0
        self.rho = options['rho']
        self.day_only = options['day_only']
        self.grad = options['grad']
        self.prob_a = options['prob_a']
        self.prob_b = options['prob_b']
        self.prob_range = self.prob_a > 0.0 and self.prob_b > 0.0
        self.ffn = self.f * self.f * self.n  # unrolled filter volume
        self.g_scale = self.wh # uniform scaling due to the "gx" function
        self.resume = resume

        if self.grad:
            self.theta = 0.0
            self.alpha = 1.0
            self.beta = 0.0
            self.lambdaM = 0.0

        self.__check_params(skip_some_checks=not options['check_params'])

        if self.wh >= 1:
            split_size = int(math.floor(self.wh / 4.0))
            self.split_sizes = [split_size, split_size, split_size, self.wh - (split_size * 3)]
        else:
            self.split_sizes = [self.wh]

        # TensorFlow session and graph
        if os.path.exists(options['root'] + '/tensor_board'):
            shutil.rmtree(options['root'] + '/tensor_board')
        self.sess = tf.Session()
        self.process_frame_ops, self.frame_0_init_op, self.saver = self.__build_graph()
        self.summary_writer = tf.summary.FileWriter(options['root'] + '/tensor_board', self.sess.graph)
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

        # quantity: 1 / delta, where delta is the ratio for computing derivatives
        if self.__first_frame:
            one_over_delta = 0.0  # so derivatives will be zero
        else:
            one_over_delta = float(self.step_size)

        # getting values that are fed to the session "runner"
        t = tf.get_default_graph().get_tensor_by_name("t:0")
        fed_frame_0 = tf.get_default_graph().get_tensor_by_name("fed_frame_0:0")
        fed_frame_1 = tf.get_default_graph().get_tensor_by_name("fed_frame_1:0")
        fed_motion_01 = tf.get_default_graph().get_tensor_by_name("fed_motion_01:0")
        fed_one_over_delta = tf.get_default_graph().get_tensor_by_name("fed_one_over_delta:0")
        feed_dict = {fed_frame_0: frame_0_to_feed,
                     fed_frame_1: frame_1_to_feed,
                     fed_motion_01: motion_01_to_feed,
                     fed_one_over_delta: one_over_delta,
                     t: (self.step - 1)}

        # fixing the case of the first frame
        if self.__first_frame:
            self.sess.run(self.frame_0_init_op, feed_dict=feed_dict)
            self.__first_frame = False

        # running the computations over the TensorFlow graph
        feature_maps, filters_matrix, \
            mi, mi_real, ce, minus_ge, sum_to_one, negativeness, motion, norm_q, norm_q_dot, norm_q_dot_dot, \
            norm_q_dot_dot_dot, \
            norm_q_mixed, all_terms, is_night, rho, \
            summary_ops, fake_op = self.sess.run(self.process_frame_ops, feed_dict=feed_dict)

        self.summary_writer.add_summary(summary_ops, self.step)

        self.step = self.step + 1

        return feature_maps, filters_matrix, \
            mi, mi_real, ce, minus_ge, sum_to_one, negativeness, motion, norm_q, norm_q_dot, norm_q_dot_dot, norm_q_dot_dot, \
            norm_q_mixed, all_terms, is_night, rho

    def __check_params(self, skip_some_checks=False):
        if self.f < 3 or self.f % 2 == 0:
            raise ValueError("Invalid filter size: " +
                             str(self.f) + "x" + str(self.f) + " (each size must be > 0 and odd)")

        if self.m < 2:
            raise ValueError("Invalid number of output features: " + str(self.m) + " (it must be >= 2)")

        if self.lambda0 < 0.0:
            raise ValueError("Invalid lambda0: " + str(self.lambda0) + " (it must be >= 0)")

        if self.lambdaE < 0.0:
            raise ValueError("Invalid lambdaE: " + str(self.lambdaE) + " (it must be >= 0)")

        if self.lambda1 < 0.0:
            raise ValueError("Invalid lambda1: " + str(self.lambda1) + " (it must be >= 0)")

        if self.lambdaC < 0.0:
            raise ValueError("Invalid lambdaC: " + str(self.lambdaC) + " (it must be >= 0)")

        if self.lambdaM < 0.0:
            raise ValueError("Invalid lambdaM: " + str(self.lambdaM) + " (it must be >= 0)")

        if not skip_some_checks:
            if self.alpha <= 0.0:
                raise ValueError("Invalid alpha: " + str(self.alpha) + " (it must be > 0)")

            if self.beta <= 0.0:
                raise ValueError("Invalid beta: " + str(self.beta) + " (it must be > 0)")

            if self.k <= 0.0:
                raise ValueError("Invalid k: " + str(self.k) + " (it must be > 0)")

            val = self.beta/self.theta
            if self.gamma <= val:
                raise ValueError("Invalid gamma: " + str(self.gamma) +
                                 " (it must be > beta/theta, where beta/theta = " + str(val) + ")")

            val = ((self.beta - (self.gamma * self.theta)) *
                   (self.beta - self.theta * (self.gamma + 2.0 * self.alpha * self.theta))) / (4.0 * self.alpha)
            if self.k >= val:
                raise ValueError("Invalid k: " + str(self.k) + " (it must be < " + str(val) + ")")

    def __build_graph(self):

        # TensorFlow precision
        precision = tf.float32

        # TensorFlow inputs
        fed_frame_0 = tf.placeholder(precision, name="fed_frame_0")
        fed_frame_1 = tf.placeholder(precision, name="fed_frame_1")
        fed_motion_01 = tf.placeholder(precision, name="fed_motion_01")
        fed_one_over_delta = tf.placeholder(precision, shape=(), name="fed_one_over_delta")
        t = tf.placeholder(precision, name="t")

        # TensorFlow variables (main scope)
        with tf.variable_scope("main", reuse=False):

            # blurring related
            tf.get_variable("rho", (), dtype=precision, initializer=tf.constant_initializer(self.rho, dtype=precision))
            tf.get_variable("night", (), dtype=precision, initializer=tf.constant_initializer(0.0, dtype=precision))
            tf.get_variable("obj_values", [12], dtype=precision, initializer=tf.constant_initializer(0.0, dtype=precision))
            tf.get_variable("step_size", (), dtype=precision, initializer=tf.constant_initializer(self.step_size, dtype=precision))

            # variables that keep what has been computed in the previous frame
            tf.get_variable("frame_0", [1, self.h, self.w, self.n], dtype=precision, initializer=tf.zeros_initializer)
            tf.get_variable("M_block_0", [self.ffn, self.ffn], dtype=precision, initializer=tf.zeros_initializer)
            tf.get_variable("N_block_0", [self.ffn, self.ffn], dtype=precision, initializer=tf.zeros_initializer)
            tf.get_variable("gradient_like_norm_0", (), dtype=precision, initializer=tf.constant_initializer(-1.0, dtype=precision))

            # the real variables (init around 0.0, with small standard deviation and cut to 2*std)
            # tf.constant_initializer(1.0)
            if not self.init_fixed:
                tf.get_variable("q1", [self.ffn, self.m], dtype=precision,
                                initializer=tf.random_uniform_initializer(-self.init_q, self.init_q))  # q
            else:
                tf.get_variable("q1", [self.ffn, self.m], dtype=precision,
                                initializer=tf.constant_initializer(self.init_q))  # q
            tf.get_variable("q2", [self.ffn, self.m], dtype=precision,
                            initializer=tf.constant_initializer(0.0))  # q^(1)
            tf.get_variable("q3", [self.ffn, self.m], dtype=precision,
                            initializer=tf.constant_initializer(0.0))  # q^(2)
            tf.get_variable("q4", [self.ffn, self.m], dtype=precision,
                            initializer=tf.constant_initializer(0.0))  # q^(3)

        # getting frames (rescaling to [0,1]) and motion (the first motion component indicates horizontal motion)
        with tf.variable_scope("main", reuse=True):
            frame_0_init_op = tf.assign(tf.get_variable("frame_0", dtype=precision),
                                        tf.expand_dims(tf.div(fed_frame_0, 255.0), 0))
            frame_0 = tf.get_variable("frame_0", dtype=precision)

        frame_1 = tf.expand_dims(tf.div(fed_frame_1, 255.0), 0)  # adding fake batch dimension 1 x h x w x n
        motion_01 = tf.expand_dims(fed_motion_01, 3)  # h x w x 2 x 1 (the 1st motion component is horizontal motion)

        # blurring
        with tf.variable_scope("main", reuse=True):
            norm_q = tf.reduce_sum(tf.square(tf.get_variable("q1", dtype=precision)))
            norm_q_dot = tf.reduce_sum(tf.square(tf.get_variable("q2", dtype=precision)))
            norm_q_dot_dot = tf.reduce_sum(tf.square(tf.get_variable("q3", dtype=precision)))
            norm_q_dot_dot_dot = tf.reduce_sum(tf.square(tf.get_variable("q4", dtype=precision)))
            norm_q_mixed = tf.reduce_sum(tf.multiply(tf.get_variable("q2", dtype=precision),
                                                     tf.get_variable("q3", dtype=precision)))

            is_night = tf.get_variable("night", dtype=precision)
            is_day = tf.abs(is_night - 1.0)

            condition1 = tf.cast(tf.less(norm_q_dot, self.eps1), precision) * \
                tf.cast(tf.less(norm_q_dot_dot, self.eps2), precision) * \
                tf.cast(tf.less(norm_q_dot_dot_dot, self.eps3), precision)

            condition2 = tf.cast(tf.less(norm_q_dot, self.eps1 * self.zeta), precision) * \
                tf.cast(tf.less(norm_q_dot_dot, self.eps2 * self.zeta), precision) * \
                tf.cast(tf.less(norm_q_dot_dot_dot, self.eps3 * self.zeta), precision)

            if not self.day_only:
                it_will_be_night = is_day * (1.0 - condition1) + is_night * (1.0 - condition2)
            else:
                it_will_be_night = 0.0

            frame_1 = (1.0 - it_will_be_night) * tf.get_variable("rho", dtype=precision) * frame_1

        # getting the spatial gradient (h x w x 2 x n (the first spatial component is horizontal))
        spatial_gradient = FeatureExtractor.__spatial_gradient(frame_1, self.h, self.w, self.n)  # frame 1 here

        # mixing the spatial gradient with motion (element-wise product + sum): h x w x n
        v_delta_gamma = tf.reduce_sum(tf.multiply(spatial_gradient, motion_01), 2)  # broadcast (and then sum) over "n"
        v_delta_gamma = tf.expand_dims(v_delta_gamma, 0)  # 1 x h x w x n

        # derivative of the input over time
        gamma_dot = tf.multiply(tf.subtract(frame_1, frame_0), fed_one_over_delta)  # 1 x h x w x n

        # extracting patches from current frame (num_splits x wh_split x filter_volume; wh x filter_volume)
        frame_patches = self.__extract_patches(frame_1)

        # extracting patches from "gamma_dot + v_delta_gamma" (num_splits x wh_split x filter_volume)
        gamma_dot_v_delta_patches = self.__extract_patches(tf.add(gamma_dot, v_delta_gamma))

        # computing a single block of M (filter volume x filter volume)
        M_block_1 = tf.div(tf.matmul(frame_patches,
                                     frame_patches, transpose_a=True), self.g_scale)

        # computing a single block of O
        O_block = tf.div(tf.matmul(gamma_dot_v_delta_patches,
                                   gamma_dot_v_delta_patches, transpose_a=True), self.g_scale)

        # computing a single block of N
        N_block_1 = tf.div(tf.matmul(gamma_dot_v_delta_patches,
                                     frame_patches, transpose_a=True), self.g_scale)

        # computing "b" and B
        b = tf.expand_dims(tf.div(tf.reduce_sum(frame_patches, 0), self.g_scale), 1)  # filter_volume x 1
        B = tf.matmul(b, b, transpose_b=True)

        # getting the previously computed quantities
        with tf.variable_scope("main", reuse=True):
            M_block_0 = tf.get_variable("M_block_0", dtype=precision)
            N_block_0 = tf.get_variable("N_block_0", dtype=precision)

        # derivatives over time
        with tf.variable_scope("main", reuse=True):
            M_block_dot = tf.multiply(tf.subtract(M_block_1, M_block_0), fed_one_over_delta)  # filter vol x filter vol
            N_block_dot = tf.multiply(tf.subtract(N_block_1, N_block_0), fed_one_over_delta)  # filter vol x filter vol

        # convolution
        with tf.variable_scope("main", reuse=True):
            filters_matrix = tf.get_variable("q1", dtype=precision)  # filter_volume x m
            feature_maps = tf.add(tf.matmul(frame_patches, filters_matrix), 1.0 / self.m)  # wh x m
            mask_sum_z = tf.cast(tf.less(feature_maps, 0.0), precision)  # wh x m

            if not self.prob_range:
                mask = (-self.lambda0/self.alpha) * mask_sum_z
            else:
                sum_feature_maps = tf.expand_dims(tf.reduce_sum(feature_maps, 1), 1)  # wh x 1
                mask_sum_a = tf.cast(tf.less(sum_feature_maps, self.prob_a), precision)  # wh x 1
                mask_sum_b = tf.cast(tf.greater(sum_feature_maps, self.prob_b), precision)  # wh x 1
                mask_sum_z = tf.cast(tf.less(feature_maps, 0.0), precision) # wh x m
                mask = (self.lambda1/self.alpha) * tf.tile((mask_sum_b - mask_sum_a), [1, self.m]) \
                       - (self.lambda0/self.alpha) * mask_sum_z

        # objective function terms
        with tf.variable_scope("main", reuse=True):
            ce = -tf.div(tf.reduce_sum(tf.square(feature_maps)), self.g_scale)
            minus_ge = tf.div(tf.reduce_sum(tf.square(tf.reduce_sum(feature_maps, 0))), self.g_scale * self.g_scale)
            mi = - ce - minus_ge

            min_f = tf.reduce_min(feature_maps, 1)
            fp = tf.subtract(feature_maps, tf.expand_dims(tf.multiply(min_f, tf.cast(tf.less(min_f, 0.0), precision)), 1))
            p = tf.maximum(tf.div(fp, tf.expand_dims(tf.reduce_sum(fp, 1), 1)), 0.00001)  # wh x m
            log_p = tf.div(tf.log(p), np.log(self.m))  # wh x m
            p_log_p = tf.multiply(p, log_p)  # wh x m
            avg_p = tf.reduce_mean(p, 0)  # m
            mi_real = tf.reduce_sum(tf.reduce_mean(p_log_p, 0)) - tf.reduce_sum(tf.multiply(avg_p, tf.div(tf.log(avg_p), np.log(self.m))))

            if not self.prob_range:
                sum_to_one = tf.div(tf.reduce_sum(tf.square(tf.reduce_sum(feature_maps, 1) - 1.0)), self.g_scale)
            else:
                sum_to_one = tf.div(tf.reduce_sum(
                        tf.multiply(mask_sum_b, sum_feature_maps - self.prob_b)
                        - tf.multiply(mask_sum_a, sum_feature_maps - self.prob_a)), self.g_scale)

            negativeness = -tf.div(tf.reduce_sum(tf.multiply(feature_maps, mask_sum_z)), self.g_scale)
            motion = tf.reduce_sum(tf.multiply(tf.get_variable("q2", dtype=precision),
                                               tf.matmul(M_block_1, tf.get_variable("q2", dtype=precision)))) \
                     + 2.0 * tf.reduce_sum(tf.multiply(tf.get_variable("q1", dtype=precision),
                                                       tf.matmul(N_block_1, tf.get_variable("q2", dtype=precision)))) \
                     + tf.reduce_sum(tf.multiply(tf.get_variable("q1", dtype=precision),
                                                 tf.matmul(O_block, tf.get_variable("q1", dtype=precision))))
            all_terms = self.lambdaC * ce + self.lambdaE * minus_ge + self.lambda0 * negativeness \
                  + self.lambda1 * sum_to_one + self.lambdaM * motion \
                  + self.alpha * norm_q_dot_dot + self.beta * norm_q_dot + self.gamma * norm_q_mixed + self.k * norm_q

            tf.summary.scalar('rho', tf.get_variable("rho", dtype=precision))
            tf.summary.scalar('MI', mi)
            tf.summary.scalar('MI_REAL', mi_real)
            tf.summary.scalar('CE', ce)
            tf.summary.scalar('-GE', minus_ge)
            tf.summary.scalar('SumToOne', sum_to_one)
            tf.summary.scalar('Negativeness', negativeness)
            tf.summary.scalar('Motion', motion)
            tf.summary.scalar('NormQ', norm_q)
            tf.summary.scalar("NormQDot", norm_q_dot)
            tf.summary.scalar("NormQDotDot", norm_q_dot_dot)
            tf.summary.scalar("NormQDotDotDot", norm_q_dot_dot_dot)
            tf.summary.scalar("QDotQDotDot", norm_q_mixed)
            tf.summary.scalar("isNight", it_will_be_night)
            tf.summary.scalar("All", all_terms)

            obj_values_normalized = tf.assign(tf.get_variable("obj_values"),
                                              tf.multiply(tf.get_variable("obj_values"), 0.9)
                                              + tf.multiply([ce, minus_ge, mi, mi_real, sum_to_one,
                                                             negativeness, motion, norm_q,
                                                             norm_q_dot, norm_q_dot_dot,
                                                             norm_q_mixed, all_terms], 0.1))

            if 1 == 2:
                ce = obj_values_normalized[0]
                minus_ge = obj_values_normalized[1]
                mi = obj_values_normalized[2]
                mi_real = obj_values_normalized[3]
                sum_to_one = obj_values_normalized[4]
                negativeness = obj_values_normalized[5]
                motion = obj_values_normalized[6]
                norm_q = obj_values_normalized[7]
                norm_q_dot = obj_values_normalized[8]
                norm_q_dot_dot = obj_values_normalized[9]
                norm_q_mixed = obj_values_normalized[10]
                all_terms = obj_values_normalized[11]

        # masked derivative of the term "w_s" (frame_patches is: wh x filter_volume)
        nab_ws = tf.div(tf.matmul(frame_patches, mask, transpose_a=True), self.g_scale)  # filter_volume x m

        # intermediate terms
        with tf.variable_scope("main", reuse=True):
            M_block_1_q1 = tf.matmul(M_block_1, tf.get_variable("q1", dtype=precision))  # filter_volume x m
            N_block_1_trans = tf.transpose(N_block_1) # filter_volume x filter_volume

        # checking constants
        assert (not np.isnan((self.k / self.alpha))) and (np.isfinite((self.k / self.alpha)))
        assert (not np.isnan((self.lambdaM * self.theta) / self.alpha)) and \
               (np.isfinite((self.lambdaM * self.theta) / self.alpha))
        assert (not np.isnan(self.lambdaM / self.alpha)) and (np.isfinite(self.lambdaM / self.alpha))
        assert (not np.isnan(1.0 / self.alpha)) and (np.isfinite(1.0 / self.alpha))
        assert (not np.isnan(self.lambda0 / self.alpha)) and (np.isfinite(self.lambda0 / self.alpha))
        assert (not np.isnan(self.lambdaM / self.alpha)) and (np.isfinite(self.lambdaM / self.alpha))
        assert (not np.isnan(self.lambda1 / self.alpha)) and (np.isfinite(self.lambda1 / self.alpha))
        assert (not np.isnan((self.gamma / self.alpha) * self.theta * self.theta -
                             (self.beta / self.alpha) * self.theta - (self.lambdaM / self.alpha) * self.theta)) and \
               (np.isfinite((self.gamma / self.alpha) * self.theta * self.theta
                            - (self.beta / self.alpha) * self.theta - (self.lambdaM / self.alpha) * self.theta))
        assert (not np.isnan(self.theta * self.theta + (self.gamma / self.alpha) * self.theta
                             - (self.beta / self.alpha))) and \
               (np.isfinite(self.theta * self.theta + (self.gamma / self.alpha) * self.theta
                            - (self.beta / self.alpha)))
        assert (not np.isnan(2 * self.theta)) and (np.isfinite(2 * self.theta))
        assert (not np.isnan((1.0 - self.lambdaC) / self.m)) and (np.isfinite((1.0 - self.lambdaC) / self.m))

        # D (this is just a portion of the D matrix in the paper)
        D = tf.multiply(tf.eye(self.ffn), self.k / self.alpha) \
            - tf.multiply(N_block_1_trans, (self.lambdaM * self.theta) / self.alpha) \
            + tf.multiply(tf.subtract(O_block, tf.transpose(N_block_dot)), self.lambdaM / self.alpha) \
            + tf.multiply(B, self.lambdaE / self.alpha)

        with tf.variable_scope("main", reuse=True):
            D_q1 = tf.matmul(D, tf.get_variable("q1", dtype=precision)) \
                - tf.multiply(M_block_1_q1, self.lambdaC / self.alpha)

            if not self.prob_range:
                D_q1 = D_q1 + tf.tile(tf.multiply(tf.expand_dims(tf.reduce_sum(M_block_1_q1, 1), 1),
                                      self.lambda1 / self.alpha), [1, self.m])

        # C
        C = tf.multiply(tf.eye(self.ffn), (self.gamma / self.alpha) * self.theta * self.theta
                        - (self.beta / self.alpha) * self.theta
                        - tf.multiply(M_block_1, (self.lambdaM / self.alpha) * self.theta)) \
            - tf.multiply(M_block_dot + N_block_1_trans - N_block_1, self.lambdaM / self.alpha)

        with tf.variable_scope("main", reuse=True):
            C_q2 = tf.matmul(C, tf.get_variable("q2", dtype=precision))

        # B
        B = tf.multiply(tf.eye(self.ffn), self.theta * self.theta
                        + (self.gamma / self.alpha) * self.theta
                        - (self.beta / self.alpha)) \
            - tf.multiply(M_block_1, self.lambdaM / self.alpha)

        with tf.variable_scope("main", reuse=True):
            B_q3 = tf.matmul(B, tf.get_variable("q3", dtype=precision))

        # A
        with tf.variable_scope("main", reuse=True):
            A_q4 = tf.multiply(tf.get_variable("q4", dtype=precision), 2 * self.theta)

        # F
        F = tf.multiply(b, (self.lambdaE - self.lambdaC) / (self.m * self.alpha)) + nab_ws

        # the real update step
        with tf.variable_scope("main", reuse=True):

            # temporarily computing the updated version of q4 (saved into another memory area)
            step_size = tf.get_variable("step_size", dtype=precision)
            gradient_like = D_q1 + C_q2 + B_q3 + A_q4 + F

            if self.step_adapt:
                gradient_like_norm_1 = tf.norm(gradient_like)

                increase = tf.cast(tf.less(gradient_like_norm_1,
                                           tf.get_variable("gradient_like_norm_0", dtype=precision)), precision)
                reduce = 1.0 - increase
                up_step = tf.assign(tf.get_variable("step_size", dtype=precision),
                                    tf.maximum(step_size * 0.1 * reduce + step_size * 2.0 * increase, self.step_size))
                with tf.control_dependencies([up_step]):
                    up_like = tf.assign(tf.get_variable("gradient_like_norm_0", dtype=precision), gradient_like_norm_1)
            else:
                up_like = tf.eye(1)

            with tf.control_dependencies([up_like]):
                if not self.grad:
                    __updated_q4 = tf.subtract(tf.get_variable("q4", dtype=precision),
                                               tf.multiply(gradient_like, tf.get_variable("step_size", dtype=precision)))
                else:
                    __updated_q4 = tf.subtract(tf.get_variable("q1", dtype=precision),
                                               tf.multiply(gradient_like, tf.get_variable("step_size", dtype=precision)))

            # updating q1, q2, q3, and q4 (the last one is updated using the temporarily computed q4)
            with tf.control_dependencies([__updated_q4]):
                up_q1 = tf.assign_add(tf.get_variable("q1", dtype=precision),
                                      tf.multiply(tf.get_variable("q2"), tf.get_variable("step_size", dtype=precision)))
                with tf.control_dependencies([up_q1]):
                    up_q2 = tf.assign_add(tf.get_variable("q2", dtype=precision),
                                          tf.multiply(tf.get_variable("q3", dtype=precision), tf.get_variable("step_size", dtype=precision)))
                    with tf.control_dependencies([up_q2]):
                        up_q3 = tf.assign_add(tf.get_variable("q3", dtype=precision),
                                              tf.multiply(tf.get_variable("q4", dtype=precision), tf.get_variable("step_size", dtype=precision)))
                        with tf.control_dependencies([up_q3]):
                            if not self.grad:
                                up_q4 = tf.assign(tf.get_variable("q4", dtype=precision), __updated_q4)
                            else:
                                up_q4 = tf.assign(tf.get_variable("q1", dtype=precision), __updated_q4)

        # updating cyclic dependencies
        with tf.control_dependencies([gamma_dot, M_block_dot, N_block_dot]):
            with tf.variable_scope("main", reuse=True):
                up_frame_0 = tf.assign(tf.get_variable("frame_0", dtype=precision), frame_1)
                up_M_block_0 = tf.assign(tf.get_variable("M_block_0", dtype=precision), M_block_1)
                up_N_block_0 = tf.assign(tf.get_variable("N_block_0", dtype=precision), N_block_1)

                diff_rho = 1.0 - tf.get_variable("rho", dtype=precision)
                up_night = tf.assign(tf.get_variable("night", dtype=precision), it_will_be_night)
                up_rho = tf.assign_add(tf.get_variable("rho", dtype=precision),
                                       self.eta * tf.cast(tf.greater(diff_rho, 0.0), precision)
                                       * diff_rho * (1.0 - it_will_be_night))

        # coordinator
        with tf.control_dependencies([up_q4, up_frame_0, up_M_block_0, up_N_block_0, up_rho, up_night]):
            fake_op = tf.eye(1)

        # operations to be executed in the data flow graph (filters_matrix: filter_volume x m)
        out_feature_maps = tf.reshape(feature_maps, [self.h, self.w, self.m])  # h x w x m
        out_filters_map = tf.transpose(tf.reshape(filters_matrix, [self.f * self.f, self.n, self.m]))  # m x n x f^2 (where each filter has been reversed)

        # summaries
        summary_ops = tf.summary.merge_all()

        ops = [out_feature_maps,
               out_filters_map,
               mi, mi_real, ce, minus_ge, sum_to_one, negativeness, motion, norm_q, norm_q_dot, norm_q_dot_dot,
               norm_q_dot_dot,
               norm_q_mixed, all_terms,
               up_night,
               up_rho,
               summary_ops,
               fake_op]

        # initialization
        if not self.resume:
            self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        return ops, frame_0_init_op, saver

    @staticmethod
    def __custom_block_matrix(matrix_block, d, around=0, around_block=None):
        matrix_shape = tf.shape(matrix_block)
        if around_block is None:
            around_block = tf.fill(matrix_shape, around)

        data_to_concat = []
        for i in range(0,d):
            data_to_concat_inner = []
            for j in range(0, d):
                if j != i:
                    data_to_concat_inner.append(around_block)
                else:
                    data_to_concat_inner.append(matrix_block)
            data_to_concat.append(tf.concat(data_to_concat_inner, 1))

        blocked = tf.concat(data_to_concat, 0)
        return blocked

    @staticmethod
    def __spatial_gradient(source, h, w, n):
        up_down_filter = tf.reshape(tf.constant([-1, 0, +1], source.dtype), [3, 1, 1, 1])
        left_right_filter = tf.reshape(up_down_filter, [1, 3, 1, 1])

        input_t = tf.transpose(source)  # n x w x h x 1

        # n x w x h x 1 (each)
        up_down_grad = tf.nn.conv2d(input_t, up_down_filter, strides=[1, 1, 1, 1], padding='SAME')
        left_right_grad = tf.nn.conv2d(input_t, left_right_filter, strides=[1, 1, 1, 1], padding='SAME')

        # returns: 1 x h x w x 2 x n (the first spatial component - 4th axis - is horizontal)
        return tf.reshape(tf.transpose(tf.stack([left_right_grad, up_down_grad], axis=1)), [h, w, 2, n])

    # computing a single block of M-like matrices (i.e., the quantity that does not depend on the output features)
    def __build_usual_matrix_block(self, unrolled_patches_splits_1, unrolled_patches_splits_2):
        mat_block_intermediate = [None] * len(self.split_sizes)

        for i in range(0, len(self.split_sizes)):  # handle data splits (this *should* be parallelized by TensorFlow)

            # what we do here is:
            # 0) tf.expand_dims: we replicate the input patches into two arrays, adding a fake singleton dimension
            #    to each of them and getting these sizes: wh_split x 1 x filter_volume; wh_split x filter_volume x 1
            # 1) tf.multiply: we multiply the two arrays: thanks to the fake dimensions and broadcasting skills of
            #    TensorFlow, we get the mixed products of the patch components: wh_split x (filter_volume^2)
            # 2) tf.reshape: we reshape to: wh_split x filter_volume x filter_volume
            # 3) tf.reduce_sum: we sum over the pixels, getting: filter_volume x filter_volume
            #mat_block_intermediate[i] = tf.reduce_sum(
            #    tf.reshape(
            #        tf.multiply(
            #            tf.expand_dims(unrolled_patches_splits_1[i], 1),
            #            tf.expand_dims(unrolled_patches_splits_2[i], 2)),
            #        [self.split_sizes[i], self.ffn, self.ffn]),
            #    0)  # here we compute the mixed products and we sum over pixels
            mat_block_intermediate[i] = tf.matmul(unrolled_patches_splits_1[i],
                                                  unrolled_patches_splits_2[i], transpose_a=True)

        mat_block = mat_block_intermediate[0]
        for i in range(1, len(self.split_sizes)):
            mat_block = tf.add(mat_block, mat_block_intermediate[i])

        return tf.div(mat_block, self.g_scale)

    # what we do here is:
    # 0) data: we are given the frame (or similar stuff) matrix: 1 x h x w x n
    # 1) tf.extract_image_patches: we extract all the patches of the same size of the filter (full filter volume)
    # 2) tf.reshape: we reshape the result to make it more manageable
    # 3) tf.split: we split the data into chunks
    def __extract_patches_splits(self, data):
        return tf.split(tf.reshape(tf.extract_image_patches(data,  # frame (or something like it) goes here
                                                            ksizes=[1, self.f, self.f, 1],
                                                            strides=[1, 1, 1, 1],
                                                            rates=[1, 1, 1, 1],
                                                            padding='SAME'),
                                   [self.wh, self.ffn]),
                        self.split_sizes, 0)  # num_splits x wh_split x filter_volume

    def __extract_patches_splits_and_also_return_full_data(self, data):
        patches = tf.reshape(tf.extract_image_patches(data,
                                                      ksizes=[1, self.f, self.f, 1],
                                                      strides=[1, 1, 1, 1],
                                                      rates=[1, 1, 1, 1],
                                                      padding='SAME'), [self.wh, self.ffn])
        return tf.split(patches, self.split_sizes, 0), patches

    def __extract_patches(self, data):
        return tf.reshape(tf.extract_image_patches(data,
                                                      ksizes=[1, self.f, self.f, 1],
                                                      strides=[1, 1, 1, 1],
                                                      rates=[1, 1, 1, 1],
                                                      padding='SAME'), [self.wh, self.ffn])


