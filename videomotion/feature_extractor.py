import tensorflow as tf
import numpy as np
import os
import shutil
# python vprocessor.py --run ../data/skater.avi --out exp/skater1 --gray 1 --save_scores_only 0 --res 240x180 --day_only 1 --rho 1.0 --check_params 0 --rep 100 --theta 0.0 --beta 1.0 --gamma 1.0 --alpha 1.0 --eta 1.0 --eps1 10000 --eps2 10000 --eps3 10000 --all_black 0 --grad 1 --m 3 --f 3 --init_q 1.0 --k 0.000001 --lambda1 0.0 --lambda0 0.0 --lambdaM 0.0 --lambdaE 2.0 --lambdaC 1.0 --step_size 0.1 --step_adapt 1 --softmax 1 --port 8888 --gew 1.0


class FeatureExtractor:

    def __init__(self, w, h, options, resume=False):
        self.__first_frame = True
        self.step = 1
        self._moving_avg_obj = -1.0

        # saving options (and other useful values) to direct attributes
        self.w = w
        self.h = h
        self.wh = w * h  # number of pixels (per input channel)
        self.step_size = options['step_size']
        self.step_size_night = options['step_size_night']
        self.step_adapt = options['step_adapt']
        self.f = options['f']  # full edge of the filter (i.e., 3 in case of 3x3 filters)
        self.n = options['n']  # input channels/features
        self.m = options['m']  # output features
        self.ffn = self.f * self.f * self.n  # unrolled filter volume
        self.init_q = options['init_q']
        self.alpha = options['alpha']
        self.beta = options['beta']
        self.theta = options['theta']
        self.thetanight = options['thetanight']
        self.k = options['k']
        self.softmax = options['softmax']
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
        self.gew = options['gew']
        self.eta = options['eta']
        self.all_black = options['all_black']
        self.init_fixed = options['init_fixed'] > 0
        self.rho = options['rho']
        self.day_only = options['day_only']
        self.grad = options['grad']
        self.prob_a = options['prob_a']
        self.prob_b = options['prob_b']
        self.shannon = options['shannon']
        self.prob_range = self.prob_a > 0.0 and self.prob_b > 0.0
        self.resume = resume

        # attention function
        self.g_scale = self.wh  # uniform scaling due to the "gx" function

        # in case of gradient-like optimization, disable some terms by zeroing their coefficients
        if self.grad:
            self.theta = 0.0  # here
            self.thetanight = 0.0  # here
            self.alpha = 1.0
            self.beta = 0.0
            self.lambdaM = 0.0

        self.__check_params(skip_some_checks=not options['check_params'])

        # TensorFlow session and graph
        self.sess = tf.Session()
        self.process_frame_ops, self.frame_0_init_op, self.saver = self.__build_graph()

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

        # quantity: 1 / delta, where delta is the ratio for computing derivatives
        if self.__first_frame:
            one_over_delta = 0.0  # so derivatives will be zero
        else:
            one_over_delta = float(self.step_size)

        # getting values that are fed to the session "runner"
        fed_frame_0 = tf.get_default_graph().get_tensor_by_name("fed_frame_0:0")
        fed_frame_1 = tf.get_default_graph().get_tensor_by_name("fed_frame_1:0")
        fed_motion_01 = tf.get_default_graph().get_tensor_by_name("fed_motion_01:0")
        fed_one_over_delta = tf.get_default_graph().get_tensor_by_name("fed_one_over_delta:0")
        feed_dict = {fed_frame_0: frame_0_to_feed,
                     fed_frame_1: frame_1_to_feed,
                     fed_motion_01: motion_01_to_feed,
                     fed_one_over_delta: one_over_delta}

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

        # TensorBoard-related
        self.summary_writer.add_summary(summary_ops, self.step)

        # next step
        self.step = self.step + 1

        # returning data (no output printing in this class, please!)
        return feature_maps, filters_matrix, \
            mi, mi_real, ce, minus_ge, sum_to_one, negativeness, motion, norm_q, norm_q_dot, norm_q_dot_dot, \
            norm_q_dot_dot, norm_q_mixed, all_terms, is_night, rho

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

            val = self.beta / self.thetanight
            if self.gamma <= val:
                raise ValueError("Invalid gamma: " + str(self.gamma) +
                                 " (it must be > beta/thetanight, where beta/thetanight = " + str(val) + ")")

            val = ((self.beta - (self.gamma * self.thetanight)) *
                   (self.beta - self.thetanight * (self.gamma + 2.0 * self.alpha * self.thetanight))) / (4.0 * self.alpha)
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

        # TensorFlow variables (main scope)
        with tf.variable_scope("main", reuse=False):

            # "blurring"-related
            rho = tf.get_variable("rho", (), dtype=precision,
                                  initializer=tf.constant_initializer(self.rho, dtype=precision))
            is_night = tf.get_variable("night", (), dtype=precision,
                                       initializer=tf.constant_initializer(0.0, dtype=precision))

            # variables that store what has been computed in the previous frame
            frame_0 = tf.get_variable("frame_0", [1, self.h, self.w, self.n], dtype=precision,
                                      initializer=tf.zeros_initializer)
            M_block_0 = tf.get_variable("M_block_0", [self.ffn, self.ffn], dtype=precision,
                                        initializer=tf.zeros_initializer)
            N_block_0 = tf.get_variable("N_block_0", [self.ffn, self.ffn], dtype=precision,
                                        initializer=tf.zeros_initializer)
            gradient_like1_0 = tf.get_variable("gradient_like1_0", [self.ffn, self.m], dtype=precision,
                                               initializer=tf.constant_initializer(-1.0, dtype=precision))
            gradient_like2_0 = tf.get_variable("gradient_like2_0", [self.ffn, self.m], dtype=precision,
                                               initializer=tf.constant_initializer(-1.0, dtype=precision))
            gradient_like3_0 = tf.get_variable("gradient_like3_0", [self.ffn, self.m], dtype=precision,
                                               initializer=tf.constant_initializer(-1.0, dtype=precision))
            gradient_like4_0 = tf.get_variable("gradient_like4_0", [self.ffn, self.m], dtype=precision,
                                               initializer=tf.constant_initializer(-1.0, dtype=precision))
            feature_map_stats = tf.get_variable("feature_map_stats", [self.wh, self.m], dtype=precision,
                                                initializer=tf.constant_initializer(1.0 / self.m, dtype=precision))

            # the real variables
            if not self.init_fixed:
                q1 = tf.get_variable("q1", [self.ffn, self.m], dtype=precision,
                                     initializer=tf.random_uniform_initializer(-self.init_q, self.init_q))  # q
            else:
                q1 = tf.get_variable("q1", [self.ffn, self.m], dtype=precision,
                                     initializer=tf.constant_initializer(self.init_q))  # q

            q2 = tf.get_variable("q2", [self.ffn, self.m], dtype=precision,
                                 initializer=tf.constant_initializer(0.0))  # q^(1)
            q3 = tf.get_variable("q3", [self.ffn, self.m], dtype=precision,
                                 initializer=tf.constant_initializer(0.0))  # q^(2)
            q4 = tf.get_variable("q4", [self.ffn, self.m], dtype=precision,
                                 initializer=tf.constant_initializer(0.0))  # q^(3)

            # getting frames (rescaling to [0,1]) and motion (the first motion component indicates horizontal motion)
            frame_0_init_op = tf.assign(frame_0, tf.expand_dims(tf.div(fed_frame_0, 255.0), 0))
            frame_1 = tf.expand_dims(tf.div(fed_frame_1, 255.0), 0)  # adding fake batch dimension 1 x h x w x n
            motion_01 = tf.expand_dims(fed_motion_01, 3)  # h x w x 2 x 1 (the 1st motion comp. is horizontal motion)

            # computing norm of variables
            norm_q = tf.reduce_sum(tf.square(q1))
            norm_q_dot = tf.reduce_sum(tf.square(q2))
            norm_q_dot_dot = tf.reduce_sum(tf.square(q3))
            norm_q_dot_dot_dot = tf.reduce_sum(tf.square(q4))
            norm_q_mixed = tf.reduce_sum(tf.multiply(q2, q3))

            # checking day and night conditions
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

            # Using the right theta and step-size
            if it_will_be_night == 1.0:
                TH = self.thetanight
                STEP = self.step_size_night
            else:
                TH = self.theta
                STEP = self.step_size

            # "optimization"-related
            obj_values = tf.get_variable("obj_values", [12], dtype=precision,
                                         initializer=tf.constant_initializer(0.0, dtype=precision))
            step_size1 = tf.get_variable("step_size1", [self.ffn, self.m], dtype=precision,
                                         initializer=tf.constant_initializer(STEP, dtype=precision))
            step_size2 = tf.get_variable("step_size2", [self.ffn, self.m], dtype=precision,
                                         initializer=tf.constant_initializer(STEP, dtype=precision))
            step_size3 = tf.get_variable("step_size3", [self.ffn, self.m], dtype=precision,
                                         initializer=tf.constant_initializer(STEP, dtype=precision))
            step_size4 = tf.get_variable("step_size4", [self.ffn, self.m], dtype=precision,
                                         initializer=tf.constant_initializer(STEP, dtype=precision))

            # blurring
            frame_1 = (1.0 - it_will_be_night) * rho * frame_1

            # getting the spatial gradient (h x w x 2 x n (the first spatial component is horizontal))
            spatial_gradient = tf.cast(
                FeatureExtractor.__spatial_gradient(tf.cast(frame_1, tf.float32),
                                                    self.h, self.w, self.n), precision)  # frame 1 here

            # mixing the spatial gradient with motion (element-wise product + sum): h x w x n
            v_delta_gamma = tf.reduce_sum(tf.multiply(spatial_gradient,
                                                      motion_01), 2)  # broadcast (and then sum) over "n"
            v_delta_gamma = tf.expand_dims(v_delta_gamma, 0)  # 1 x h x w x n

            # derivative of the input over time (fed_one_over_delta = 0 when t = 0)
            gamma_dot = tf.multiply(tf.subtract(frame_1, frame_0), fed_one_over_delta)  # 1 x h x w x n

            # extracting patches from current frame (wh x filter_volume)
            frame_patches = self.__extract_patches(frame_1)

            # do we want to squash the almost constant patches?
            # mean, var = tf.nn.moments(frame_patches, axes=1)
            # mask_var = tf.expand_dims(tf.cast(tf.greater(var, 0.001), precision), 1)
            # frame_patches = tf.multiply(frame_patches, mask_var)

            # extracting patches from "gamma_dot + v_delta_gamma" (wh x filter_volume)
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

            # other derivatives over time (fed_one_over_delta = 0 when t = 0)
            M_block_dot = tf.multiply(tf.subtract(M_block_1, M_block_0), fed_one_over_delta)  # filter vol x filter vol
            N_block_dot = tf.multiply(tf.subtract(N_block_1, N_block_0), fed_one_over_delta)  # filter vol x filter vol

            if self.softmax:
                # convolution
                feature_maps = tf.nn.softmax(tf.matmul(frame_patches, q1), dim=1)  # wh x m

                # objective function terms: ce, -ge, mi
                if self.shannon:
                    p = tf.maximum(feature_maps, 0.00001)
                    p_log_p = tf.multiply(p, tf.div(tf.log(p), np.log(self.m)))  # wh x m
                    biased_p = self.gew * p + (1.0 - self.gew) * feature_map_stats
                    avg_p = tf.reduce_mean(biased_p, 0)  # m
                    ce = -tf.reduce_sum(tf.reduce_mean(p_log_p, 0))
                    minus_ge = tf.reduce_sum(tf.multiply(avg_p, tf.div(tf.log(avg_p), np.log(self.m))))
                    mi = - ce - minus_ge
                else:
                    ce = -tf.div(tf.reduce_sum(tf.square(feature_maps)), self.g_scale)
                    minus_ge = tf.div(tf.reduce_sum(tf.square(tf.reduce_sum(feature_maps, 0))),
                                      self.g_scale * self.g_scale)
                    mi = - ce - minus_ge

                # objective function terms: probabilistic constraints
                sum_to_one = tf.cast(tf.identity(0.0), precision)
                negativeness = tf.cast(tf.identity(0.0), precision)
            else:
                # convolution
                feature_maps = tf.add(tf.matmul(frame_patches, q1), 1.0 / self.m)  # wh x m

                # objective function terms: ce, -ge, mi
                ce = -tf.div(tf.reduce_sum(tf.square(feature_maps)), self.g_scale)
                minus_ge = tf.div(tf.reduce_sum(tf.square(tf.reduce_sum(feature_maps, 0))),
                                  self.g_scale * self.g_scale)
                mi = - ce - minus_ge

                # masks for piecewise-linear constraints
                if not self.prob_range:
                    sum_feature_maps = None
                    mask_sum_a = None
                    mask_sum_b = None
                    mask_sum_z = tf.cast(tf.less(feature_maps, 0.0), precision)  # wh x m
                    mask = (-self.lambda0 / self.alpha) * mask_sum_z
                else:
                    sum_feature_maps = tf.expand_dims(tf.reduce_sum(feature_maps, 1), 1)  # wh x 1
                    mask_sum_a = tf.cast(tf.less(sum_feature_maps, self.prob_a), precision)  # wh x 1
                    mask_sum_b = tf.cast(tf.greater(sum_feature_maps, self.prob_b), precision)  # wh x 1
                    mask_sum_z = tf.cast(tf.less(feature_maps, 0.0), precision)  # wh x m
                    mask = (self.lambda1 / self.alpha) * tf.tile((mask_sum_b - mask_sum_a), [1, self.m]) \
                           - (self.lambda0 / self.alpha) * mask_sum_z

                # objective function terms: probabilistic constraints
                if not self.prob_range:
                    sum_to_one = 0.5 * tf.div(tf.reduce_sum(tf.square(tf.reduce_sum(feature_maps, 1) - 1.0)), self.g_scale)
                else:
                    sum_to_one = tf.div(tf.reduce_sum(
                            tf.multiply(mask_sum_b, sum_feature_maps - self.prob_b)
                            - tf.multiply(mask_sum_a, sum_feature_maps - self.prob_a)), self.g_scale)

                negativeness = -tf.div(tf.reduce_sum(tf.multiply(feature_maps, mask_sum_z)), self.g_scale)

            # objective function terms: motion
            motion = tf.reduce_sum(tf.multiply(q2, tf.matmul(M_block_1, q2))) \
                + 2.0 * tf.reduce_sum(tf.multiply(q1, tf.matmul(N_block_1, q2))) \
                + tf.reduce_sum(tf.multiply(q1, tf.matmul(O_block, q1)))

            # objective function
            obj = self.lambdaC * 0.5 * ce + self.lambdaE * 0.5 * minus_ge + self.lambda0 * negativeness \
                + self.lambda1 * sum_to_one + self.lambdaM * motion \
                + self.alpha * norm_q_dot_dot + self.beta * norm_q_dot \
                + self.gamma * norm_q_mixed + self.k * norm_q

            # real mutual information
            min_f = tf.reduce_min(feature_maps, 1)
            fp = tf.subtract(feature_maps,
                             tf.expand_dims(tf.multiply(min_f, tf.cast(tf.less(min_f, 0.0), precision)), 1))
            p = tf.maximum(tf.div(fp, tf.expand_dims(tf.reduce_sum(fp, 1), 1)), 0.00001)  # wh x m
            log_p = tf.div(tf.log(p), np.log(self.m))  # wh x m
            p_log_p = tf.multiply(p, log_p)  # wh x m
            avg_p = tf.reduce_mean(p, 0)  # m
            mi_real = tf.reduce_sum(tf.reduce_mean(p_log_p, 0)) - \
                tf.reduce_sum(tf.multiply(avg_p, tf.div(tf.log(avg_p), np.log(self.m))))

            # TensorBoard-related
            tf.summary.scalar('J_Rho', rho)
            tf.summary.scalar('C_MutualInformation', mi)
            tf.summary.scalar('D_RealMutualInformation', mi_real)
            tf.summary.scalar('E_ConditionalEntropy', ce)
            tf.summary.scalar('F_MinusEntropy', minus_ge)
            tf.summary.scalar('G_SumToOne', sum_to_one)
            tf.summary.scalar('H_Negativeness', negativeness)
            tf.summary.scalar('I_Motion', motion)
            tf.summary.scalar('K_NormQ', norm_q)
            tf.summary.scalar("L_NormQDot", norm_q_dot)
            tf.summary.scalar("M_NormQDotDot", norm_q_dot_dot)
            tf.summary.scalar("N_NormQDotDotDot", norm_q_dot_dot_dot)
            tf.summary.scalar("O_QDotQDotDot", norm_q_mixed)
            tf.summary.scalar("B_IsNight", it_will_be_night)
            tf.summary.scalar("A_FullObjectiveFunction", obj)

            # moving-average on objective function terms
            if self._moving_avg_obj > 0.0:
                obj_values_normalized = tf.assign(obj_values,
                                                  tf.multiply(obj_values, 1.0 - self._moving_avg_obj)
                                                  + tf.multiply([ce, minus_ge, mi, mi_real, sum_to_one,
                                                                 negativeness, motion, norm_q,
                                                                 norm_q_dot, norm_q_dot_dot,
                                                                 norm_q_mixed, obj], self._moving_avg_obj))

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
                obj = obj_values_normalized[11]

            # intermediate terms
            M_block_1_q1 = tf.matmul(M_block_1, q1)  # filter_volume x m
            N_block_1_trans = tf.transpose(N_block_1)  # filter_volume x filter_volume

            # checking constants
            assert (not np.isnan((self.k / self.alpha))) and (np.isfinite((self.k / self.alpha)))
            assert (not np.isnan((self.lambdaM * TH) / self.alpha)) and \
                   (np.isfinite((self.lambdaM * TH) / self.alpha))
            assert (not np.isnan(self.lambdaM / self.alpha)) and (np.isfinite(self.lambdaM / self.alpha))
            assert (not np.isnan(1.0 / self.alpha)) and (np.isfinite(1.0 / self.alpha))
            assert (not np.isnan(self.lambda0 / self.alpha)) and (np.isfinite(self.lambda0 / self.alpha))
            assert (not np.isnan(self.lambdaM / self.alpha)) and (np.isfinite(self.lambdaM / self.alpha))
            assert (not np.isnan(self.lambda1 / self.alpha)) and (np.isfinite(self.lambda1 / self.alpha))
            assert (not np.isnan((self.gamma / self.alpha) * TH * TH -
                                 (self.beta / self.alpha) * TH - (self.lambdaM / self.alpha) * TH)) \
                and \
                   (np.isfinite((self.gamma / self.alpha) * TH * TH
                                - (self.beta / self.alpha) * TH - (self.lambdaM / self.alpha) * TH))
            assert (not np.isnan(TH * TH + (self.gamma / self.alpha) * TH
                                 - (self.beta / self.alpha))) and \
                   (np.isfinite(TH * TH + (self.gamma / self.alpha) * TH
                                - (self.beta / self.alpha)))
            assert (not np.isnan(2 * TH)) and (np.isfinite(2 * TH))
            assert (not np.isnan((1.0 - self.lambdaC) / self.m)) and (np.isfinite((1.0 - self.lambdaC) / self.m))

            # D (this is just a portion of the D matrix in the paper)
            D = (self.k / self.alpha) * tf.cast(tf.eye(self.ffn), precision) \
                - ((self.lambdaM * TH) / self.alpha) * N_block_1_trans \
                + (self.lambdaM / self.alpha) * tf.subtract(O_block, tf.transpose(N_block_dot)) \

            if not self.softmax:
                D = D + (self.lambdaE / self.alpha) * B
                D_q1 = tf.matmul(D, q1) - (self.lambdaC / self.alpha) * M_block_1_q1

                if not self.prob_range:
                    D_q1 = D_q1 + tf.tile((self.lambda1 / self.alpha) *
                                          tf.expand_dims(tf.reduce_sum(M_block_1_q1, 1), 1), [1, self.m])
            else:
                D_q1 = tf.matmul(D, q1)

            # C
            C = (((self.gamma / self.alpha) * TH * TH) \
                - (self.beta / self.alpha) * TH) * tf.cast(tf.eye(self.ffn), precision) \
                - ((self.lambdaM / self.alpha) * TH) * M_block_1 \
                - (self.lambdaM / self.alpha) * (M_block_dot + N_block_1_trans - N_block_1)

            C_q2 = tf.matmul(C, q2)

            # B
            Bbb = (TH * TH + (self.gamma / self.alpha) * TH - (self.beta / self.alpha)) \
                  * tf.cast(tf.eye(self.ffn), precision) - (self.lambdaM / self.alpha) * M_block_1

            B_q3 = tf.matmul(Bbb, q3)

            # A
            A_q4 = tf.multiply(q4, 2.0 * TH)

            # F
            if not self.softmax:
                nab_ws = tf.div(tf.matmul(frame_patches, mask, transpose_a=True), self.g_scale)  # filter_volume x m
                F = ((self.lambdaE - self.lambdaC) / (self.m * self.alpha)) * b + nab_ws
            else:
                g = 1.0 / self.g_scale
                Sg = tf.expand_dims(tf.reduce_sum(feature_maps * g, 0), 0)
                Ss = feature_maps * feature_maps * g
                F_ge = feature_maps * (Sg * g) - tf.matmul(feature_maps, Sg * g, transpose_b=True) * feature_maps
                F_ce = -Ss + feature_maps * tf.expand_dims(tf.reduce_sum(Ss, 1), 1)
                F = tf.matmul(frame_patches, self.lambdaE * F_ge + self.lambdaC * F_ce, transpose_a=True)

            # update terms
            gradient_like1 = -q2
            gradient_like2 = -q3
            gradient_like3 = -q4
            gradient_like4 = D_q1 + C_q2 + B_q3 + A_q4 + F

            # step sizes
            if self.step_adapt:
                increase1 = tf.cast(tf.greater(gradient_like1 * gradient_like1_0, 0.0), precision)
                reduce1 = 1.0 - increase1
                step_size1 = tf.assign(step_size1,
                                       tf.minimum(tf.maximum(step_size1 * 0.1 * reduce1 + step_size1 * 2.0 * increase1,
                                                             STEP), STEP * 1000))
                increase2 = tf.cast(tf.greater(gradient_like2 * gradient_like2_0, 0.0), precision)
                reduce2 = 1.0 - increase2
                step_size2 = tf.assign(step_size2,
                                       tf.minimum(tf.maximum(step_size2 * 0.1 * reduce2 + step_size2 * 2.0 * increase2,
                                                             STEP), STEP * 1000))
                increase3 = tf.cast(tf.greater(gradient_like3 * gradient_like3_0, 0.0), precision)
                reduce3 = 1.0 - increase3
                step_size3 = tf.assign(step_size3,
                                       tf.minimum(tf.maximum(step_size3 * 0.1 * reduce3 + step_size3 * 2.0 * increase3,
                                                             STEP), STEP * 1000))
                increase4 = tf.cast(tf.greater(gradient_like4 * gradient_like4_0, 0.0), precision)
                reduce4 = 1.0 - increase4
                step_size4 = tf.assign(step_size4,
                                       tf.minimum(tf.maximum(step_size4 * 0.1 * reduce4 + step_size4 * 2.0 * increase4,
                                                             STEP), STEP * 1000))

            # update rules
            with tf.control_dependencies([step_size1, step_size2, step_size3, step_size4]):
                if not self.grad:
                    __updated_q4 = q4 - gradient_like4 * step_size4
                else:
                    up_q4 = tf.assign(q1, q1 - gradient_like4 * step_size4)

            if not self.grad:
                with tf.control_dependencies([__updated_q4]):
                    up_q1 = tf.assign_sub(q1, tf.multiply(gradient_like1, step_size1))
                    with tf.control_dependencies([up_q1]):
                        up_q2 = tf.assign_sub(q2, tf.multiply(gradient_like2, step_size2))
                        with tf.control_dependencies([up_q2]):
                            up_q3 = tf.assign_sub(q3, tf.multiply(gradient_like3, step_size3))
                            with tf.control_dependencies([up_q3]):
                                up_q4 = tf.assign(q4, __updated_q4)

            # updating cyclic dependencies
            with tf.control_dependencies([gamma_dot, M_block_dot, N_block_dot]):
                up_frame_0 = tf.assign(frame_0, frame_1)
                up_M_block_0 = tf.assign(M_block_0, M_block_1)
                up_N_block_0 = tf.assign(N_block_0, N_block_1)
                up_gradient_like1 = tf.assign(gradient_like1_0, gradient_like1)
                up_gradient_like2 = tf.assign(gradient_like2_0, gradient_like2)
                up_gradient_like3 = tf.assign(gradient_like3_0, gradient_like3)
                up_gradient_like4 = tf.assign(gradient_like4_0, gradient_like4)
                if self.softmax and self.shannon:
                    up_feature_map_stats = tf.assign(feature_map_stats, biased_p)
                else:
                    up_feature_map_stats = tf.eye(1)

                diff_rho = 1.0 - rho
                up_night = tf.assign(is_night, it_will_be_night)
                up_rho = tf.assign_add(rho,
                                       self.eta * tf.cast(tf.greater(diff_rho, 0.0), precision)
                                       * diff_rho * (1.0 - it_will_be_night))

            # coordinator
            with tf.control_dependencies([up_q4, up_frame_0, up_M_block_0, up_N_block_0,
                                          up_gradient_like1, up_gradient_like2, up_gradient_like3, up_gradient_like4,
                                          up_feature_map_stats, up_rho, up_night]):
                fake_op = tf.eye(1)

            # operations to be executed in the data flow graph (filters_matrix: filter_volume x m)
            out_feature_maps = tf.reshape(feature_maps, [self.h, self.w, self.m])  # h x w x m
            out_filters_map = tf.transpose(tf.reshape(q1, [self.f * self.f, self.n, self.m]))  # m x n x f^2

            # summaries
            summary_ops = tf.summary.merge_all()

            ops = [out_feature_maps,
                   out_filters_map,
                   mi, mi_real, ce, minus_ge, sum_to_one, negativeness, motion, norm_q, norm_q_dot, norm_q_dot_dot,
                   norm_q_dot_dot,
                   norm_q_mixed, obj,
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
    def __spatial_gradient(source, h, w, n):
        up_down_filter = tf.reshape(tf.constant([-1, 0, +1], source.dtype), [3, 1, 1, 1])
        left_right_filter = tf.reshape(up_down_filter, [1, 3, 1, 1])

        input_t = tf.transpose(source)  # n x w x h x 1

        # n x w x h x 1 (each)
        up_down_grad = tf.nn.conv2d(input_t, up_down_filter, strides=[1, 1, 1, 1], padding='SAME')
        left_right_grad = tf.nn.conv2d(input_t, left_right_filter, strides=[1, 1, 1, 1], padding='SAME')

        # returns: 1 x h x w x 2 x n (the first spatial component - 4th axis - is horizontal)
        return tf.reshape(tf.transpose(tf.stack([left_right_grad, up_down_grad], axis=1)), [h, w, 2, n])

    def __extract_patches(self, data):
        return tf.reshape(tf.extract_image_patches(data,
                                                   ksizes=[1, self.f, self.f, 1],
                                                   strides=[1, 1, 1, 1],
                                                   rates=[1, 1, 1, 1],
                                                   padding='SAME'), [self.wh, self.ffn])
