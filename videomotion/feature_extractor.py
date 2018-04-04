import tensorflow as tf
import numpy as np
import os
import shutil


class FeatureExtractor:

    def __init__(self, w, h, options, resume=False):
        self.__first_frame = True
        self.step = 1

        # saving options (and other useful values) to direct attributes
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
        self.eps1 = options['eps1'] * self.m
        self.eps2 = options['eps2'] * self.m
        self.eps3 = options['eps3'] * self.m
        self.gew = options['gew']
        self.eta = options['eta']
        self.all_black = options['all_black']
        self.init_fixed = options['init_fixed']
        self.rho = options['rho']
        self.day_only = options['day_only']
        self.grad = options['grad']
        self.root = options['root']
        self.rk = options['rk']
        self.stream = options['stream']

        # saving other parameters
        self.resume = resume
        self.summary_writer = None
        self.w = w
        self.h = h
        self.wh = w * h  # number of pixels (per input channel)
        self.ffn = self.f * self.f * self.n  # unrolled filter volume

        # attention function
        self.g_scale = float(self.wh)  # uniform scaling due to the "gx" function

        # in case of gradient-like optimization, disable some terms by zeroing their coefficients
        if self.grad:
            self.theta = 0.0  # this should >> 1
            self.alpha = 1.0  # this should be 0, but I need this value (1.0) due to implementation issues
            self.beta = 0.0
            self.gamma = 0.0  # this should be 1/(\theta^2)
            self.lambdaM = 0.0

        self.__check_params(skip_some_checks=not options['check_params'])

        # TensorFlow session and graph
        self.sess = tf.Session()
        self.process_frame_ops, self.frame_0_init_op, self.saver, self.t_reset_op = self.__build_graph()
        self.activate_tensor_board()

        # TensorFlow model, save dir
        self.save_path = options['root'] + '/model/model.saved'

    def close(self):
        self.summary_writer.close()
        self.sess.close()

    def save(self):
        self.saver.save(self.sess, self.save_path)

    def load(self, steps):
        self.saver.restore(self.sess, self.save_path)
        self.step = steps

    def activate_tensor_board(self):
        if (not self.resume) and os.path.exists(self.root + '/tensor_board'):
            shutil.rmtree(self.root + '/tensor_board')
        self.summary_writer = tf.summary.FileWriter(self.root + '/tensor_board', self.sess.graph)

    # processing the current frame (frame_1 below)
    def run_step(self, frame_1_to_feed, motion_01_to_feed):

        # zero signal
        if self.all_black > 0:
            frame_1_to_feed.fill(0.0)
            motion_01_to_feed.fill(0.0)

        # quantity: 1 / delta, where delta is the ratio for computing derivatives
        if self.__first_frame and self.step == 1:
            one_over_delta = 0.0  # so derivatives will be zero
        else:
            one_over_delta = float(self.step_size)

        # getting values that are fed to the session "runner"
        fed_frame_1 = tf.get_default_graph().get_tensor_by_name("fed_frame_1:0")
        fed_motion_01 = tf.get_default_graph().get_tensor_by_name("fed_motion_01:0")
        fed_one_over_delta = tf.get_default_graph().get_tensor_by_name("fed_one_over_delta:0")
        feed_dict = {fed_frame_1: frame_1_to_feed,
                     fed_motion_01: motion_01_to_feed,
                     fed_one_over_delta: one_over_delta}

        # fixing the case of the first frame
        if self.__first_frame:
            self.sess.run(self.frame_0_init_op, feed_dict=feed_dict)
            if self.step == 1:
                self.sess.run(self.t_reset_op, feed_dict=feed_dict)
            self.__first_frame = False

        # running the computations over the TensorFlow graph
        ret = self.sess.run(self.process_frame_ops, feed_dict=feed_dict)

        # TensorBoard-related
        self.summary_writer.add_summary(ret[-2], self.step)

        # next step
        self.step = self.step + 1

        # returning data (no output printing in this class, please!)
        return ret[0:-2]

    def __check_params(self, skip_some_checks=False):
        if self.f < 3 or self.f % 2 == 0:
            raise ValueError("Invalid filter size: " +
                             str(self.f) + "x" + str(self.f) + " (each size must be > 0 and odd)")

        if self.m < 2:
            raise ValueError("Invalid number of output features: " + str(self.m) + " (it must be >= 2)")

        if self.lambdaE < 0.0:
            raise ValueError("Invalid lambdaE: " + str(self.lambdaE) + " (it must be >= 0)")

        if self.lambdaC < 0.0:
            raise ValueError("Invalid lambdaC: " + str(self.lambdaC) + " (it must be >= 0)")

        if self.lambdaM < 0.0:
            raise ValueError("Invalid lambdaM: " + str(self.lambdaM) + " (it must be >= 0)")

        if self.eps1 < 0.0:
            raise ValueError("Invalid eps1: " + str(self.eps1) + " (it must be >= 0)")

        if self.eps2 < 0.0:
            raise ValueError("Invalid eps2: " + str(self.eps2) + " (it must be >= 0)")

        if self.eps3 < 0.0:
            raise ValueError("Invalid eps3: " + str(self.eps3) + " (it must be >= 0)")

        if not skip_some_checks:
            if self.alpha <= 0.0:
                raise ValueError("Invalid alpha_night: " + str(self.alpha) + " (it must be > 0)")

            if self.beta <= 0.0:
                raise ValueError("Invalid beta: " + str(self.beta) + " (it must be > 0)")

            if self.k <= 0.0:
                raise ValueError("Invalid k: " + str(self.k) + " (it must be > 0)")

            val = self.beta / self.theta
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
        fed_frame_1 = tf.placeholder(precision, name="fed_frame_1")
        fed_motion_01 = tf.placeholder(precision, name="fed_motion_01")
        fed_one_over_delta = tf.placeholder(precision, shape=(), name="fed_one_over_delta")

        # TensorFlow variables (main scope)
        with tf.variable_scope("main", reuse=False):
            rho = tf.get_variable("rho", (), dtype=precision,
                                  initializer=tf.constant_initializer(self.rho, dtype=precision))
            is_night = tf.get_variable("night", (), dtype=precision,
                                       initializer=tf.constant_initializer(0.0, dtype=precision))
            t = tf.get_variable("t", (), dtype=precision, initializer=tf.constant_initializer(0.0, dtype=precision))
            avg_p_full = tf.get_variable("avg_p_full", [self.m], dtype=precision,
                                         initializer=tf.constant_initializer(0.0, dtype=precision))
            avg_p_log_p_full = tf.get_variable("avg_p_log_p_full", [self.m], dtype=precision,
                                               initializer=tf.constant_initializer(0.0, dtype=precision))
            motion_full = tf.get_variable("motion_full", (), dtype=precision,
                                          initializer=tf.constant_initializer(0.0, dtype=precision))
            obj_comp_values = tf.get_variable("obj_comp_values", [12], dtype=precision,
                                              initializer=tf.constant_initializer(0.0, dtype=precision))
            step_size = tf.get_variable("step_size", [self.ffn, self.m], dtype=precision,
                                        initializer=tf.constant_initializer(self.step_size, dtype=precision))
            feature_map_stats = tf.get_variable("feature_map_stats", [self.wh, self.m], dtype=precision,
                                                initializer=tf.constant_initializer(1.0 / self.m, dtype=precision))

            # variables that store what has been computed in the previous frame
            frame_0 = tf.get_variable("frame_0", [1, self.h, self.w, self.n], dtype=precision,
                                      initializer=tf.zeros_initializer)
            M_block_0 = tf.get_variable("M_block_0", [self.ffn, self.ffn], dtype=precision,
                                        initializer=tf.zeros_initializer)
            N_block_0 = tf.get_variable("N_block_0", [self.ffn, self.ffn], dtype=precision,
                                        initializer=tf.zeros_initializer)
            if self.grad and self.step_adapt:
                gradient_like_0 = tf.get_variable("gradient_like_0", [self.ffn, self.m], dtype=precision,
                                                  initializer=tf.constant_initializer(-1.0, dtype=precision))

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
            frame_0_init_op = tf.assign(frame_0, tf.expand_dims(tf.div(fed_frame_1, 255.0), 0))
            t_reset_op = tf.assign(t, 0.0)
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

            condition = tf.cast(tf.less(norm_q_dot, self.eps1), precision) * \
                tf.cast(tf.less(norm_q_dot_dot, self.eps2), precision) * \
                tf.cast(tf.less(norm_q_dot_dot_dot, self.eps3), precision)

            if not self.day_only:
                night = is_day * (1.0 - condition) + is_night * (1.0 - condition)
            else:
                night = 0.0

            # extracting patches from current frame (wh x filter_volume)
            frame_patches = self.__extract_patches(frame_1)

            # do we want to squash the almost constant patches?
            # mean, var = tf.nn.moments(frame_patches, axes=1)
            # mask_var = tf.expand_dims(tf.cast(tf.greater(var, 0.001), precision), 1)
            # frame_patches = tf.multiply(frame_patches, mask_var)

            if self.lambdaM > 0.0:

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

                # other derivatives over time (fed_one_over_delta = 0 when t = 0)
                M_block_dot = tf.multiply(tf.subtract(M_block_1, M_block_0),
                                          fed_one_over_delta)  # filter vol x filter vol
                N_block_dot = tf.multiply(tf.subtract(N_block_1, N_block_0),
                                          fed_one_over_delta)  # filter vol x filter vol
            else:
                gamma_dot = tf.eye(1)
                M_block_dot = tf.eye(1)
                N_block_dot = tf.eye(1)

            # convolution
            feature_maps = tf.nn.softmax(tf.matmul(frame_patches, q1), dim=1)  # wh x m

            # objective function terms: ce, -ge, mi
            biased_maps = float(self.gew) * feature_maps + (1.0 - float(self.gew)) * feature_map_stats
            ce2 = -tf.div(tf.reduce_sum(tf.square(feature_maps)), self.g_scale)
            minus_ge2 = tf.div(tf.reduce_sum(tf.square(tf.reduce_sum(biased_maps, 0))), self.g_scale * self.g_scale)

            ce = ((ce2 + 1.0) * float(self.m)) / (float(self.m) - 1.0)
            minus_ge = (float(self.m) * minus_ge2 - 1.0) / (float(self.m) - 1.0)

            mi2 = - ce - minus_ge

            mi = mi2 + 1.0

            # objective function terms: motion
            if self.lambdaM > 0.0:
                motion = tf.reduce_sum(tf.multiply(q2, tf.matmul(M_block_1, q2))) \
                    + 2.0 * tf.reduce_sum(tf.multiply(q1, tf.matmul(N_block_1, q2))) \
                    + tf.reduce_sum(tf.multiply(q1, tf.matmul(O_block, q1)))
            else:
                motion = 0.0

            # objective function
            obj = self.lambdaC * 0.5 * ce2 + self.lambdaE * 0.5 * minus_ge2 + \
                + self.lambdaM * motion \
                + self.alpha * norm_q_dot_dot + self.beta * norm_q_dot + self.gamma * norm_q_mixed + self.k * norm_q

            # real mutual information
            min_f = tf.reduce_min(feature_maps, 1)
            fp = tf.subtract(feature_maps,
                             tf.expand_dims(tf.multiply(min_f, tf.cast(tf.less(min_f, 0.0), precision)), 1))
            p = tf.maximum(tf.div(fp, tf.expand_dims(tf.reduce_sum(fp, 1), 1)), 0.00001)  # wh x m
            log_p = tf.div(tf.log(p), np.log(self.m))  # wh x m
            p_log_p = tf.multiply(p, log_p)  # wh x m
            avg_p = tf.reduce_mean(p, 0)  # m
            avg_p_log_p = tf.reduce_mean(p_log_p, 0)  # m
            mi_real = tf.reduce_sum(avg_p_log_p) - \
                tf.reduce_sum(tf.multiply(avg_p, tf.div(tf.log(avg_p), np.log(self.m))))

            # updating stats
            t_update = tf.assign_add(t, 1.0)
            avg_p_full_update = tf.assign_add(avg_p_full, (avg_p - avg_p_full) / t_update)
            avg_p_log_p_full_update = tf.assign_add(avg_p_log_p_full, (avg_p_log_p - avg_p_log_p_full) / t_update)
            mi_real_full = tf.reduce_sum(avg_p_log_p_full_update) - \
                tf.reduce_sum(tf.multiply(avg_p_full_update, tf.div(tf.log(avg_p_full_update), np.log(self.m))))
            motion_full_update = tf.assign_add(motion_full, (motion - motion_full) / t_update)

            # scaling
            w = tf.exp(self.theta * t)
            scaling = obj_comp_values[0] + w

            # average on objective function terms
            obj_comp = (obj_comp_values[1] + obj * w) / scaling
            ce_comp = (obj_comp_values[2] + ce * w) / scaling
            minus_ge_comp = (obj_comp_values[3] + minus_ge * w) / scaling
            mi_comp = (obj_comp_values[4] + mi * w) / scaling
            mi_real_comp = (obj_comp_values[5] + mi_real * w) / scaling
            motion_comp = (obj_comp_values[6] + motion * w) / scaling
            norm_q_comp = (obj_comp_values[7] + norm_q * w) / scaling
            norm_q_dot_comp = (obj_comp_values[8] + norm_q_dot * w) / scaling
            norm_q_dot_dot_comp = (obj_comp_values[9] + norm_q_dot_dot * w) / scaling
            norm_q_mixed_comp = (obj_comp_values[10] + norm_q_mixed * w) / scaling
            norm_q_dot_dot_dot_comp = (obj_comp_values[11] + norm_q_dot_dot_dot * w) / scaling

            obj_values = tf.identity([1.0, obj, ce, minus_ge, mi, mi_real, motion,
                                      norm_q, norm_q_dot, norm_q_dot_dot, norm_q_mixed, norm_q_dot_dot_dot])

            obj_comp_values_updated = tf.assign(obj_comp_values, [scaling, obj_comp, ce_comp, minus_ge_comp,
                                                                  mi_comp, mi_real_comp, motion_comp, norm_q_comp,
                                                                  norm_q_dot_comp, norm_q_dot_dot_comp,
                                                                  norm_q_mixed_comp, norm_q_dot_dot_dot_comp])

            # TensorBoard-related
            tf.summary.scalar("AA_Night", night)
            tf.summary.scalar('AB_Rho', rho)
            tf.summary.scalar("AC_RealMutualInformationFull", mi_real_full)
            tf.summary.scalar("AD_MotionFull", motion_full_update)

            tf.summary.scalar("BA_CognitiveAction", obj)
            tf.summary.scalar('BB_MutualInformation', mi)
            tf.summary.scalar('BC_RealMutualInformation', mi_real)
            tf.summary.scalar('BD_ConditionalEntropy', ce)
            tf.summary.scalar('BE_MinusEntropy', minus_ge)
            tf.summary.scalar('BF_Motion', motion)
            tf.summary.scalar('BG_NormQ', norm_q)
            tf.summary.scalar("BH_NormQDot", norm_q_dot)
            tf.summary.scalar("BI_NormQDotDot", norm_q_dot_dot)
            tf.summary.scalar("BJ_NormQDot_QDotDot", norm_q_mixed)
            tf.summary.scalar("BK_NormQDotDotDot", norm_q_dot_dot_dot)

            tf.summary.scalar("CA_CompleteCognitiveAction", obj_comp)
            tf.summary.scalar('CB_CompleteMutualInformation', mi_comp)
            tf.summary.scalar('CC_CompleteRealMutualInformation', mi_real_comp)
            tf.summary.scalar('CD_CompleteConditionalEntropy', ce_comp)
            tf.summary.scalar('CE_CompleteMinusEntropy', minus_ge_comp)
            tf.summary.scalar('CF_CompleteMotion', motion_comp)
            tf.summary.scalar('CG_CompleteNormQ', norm_q_comp)
            tf.summary.scalar("CH_CompleteNormQDot", norm_q_dot_comp)
            tf.summary.scalar("CI_CompleteNormQDotDot", norm_q_dot_dot_comp)
            tf.summary.scalar("CJ_CompleteNormQDot_QDotDot", norm_q_mixed_comp)
            tf.summary.scalar("CK_CompleteNormQDotDotDot", norm_q_dot_dot_dot_comp)

            # intermediate terms
            if self.lambdaM > 0.0:
                N_block_1_trans = tf.transpose(N_block_1)  # filter_volume x filter_volume

            # D (this is just a portion of the D matrix in the paper)
            if self.lambdaM > 0.0:
                D = self.k * tf.cast(tf.eye(self.ffn), precision) \
                    - (self.lambdaM * self.theta) * N_block_1_trans \
                    + self.lambdaM * tf.subtract(O_block, tf.transpose(N_block_dot))
            else:
                D = self.k * tf.cast(tf.eye(self.ffn), precision)

            # C
            if self.lambdaM > 0.0:
                C = (self.gamma * self.theta * self.theta - self.beta * self.theta) * \
                    tf.cast(tf.eye(self.ffn), precision) \
                    - (self.lambdaM * self.theta) * M_block_1 \
                    - self.lambdaM * (M_block_dot + N_block_1_trans - N_block_1)
            else:
                C = (self.gamma * self.theta * self.theta - self.beta * self.theta) * \
                    tf.cast(tf.eye(self.ffn), precision)

            # B
            if self.lambdaM > 0.0:
                B = (self.alpha * self.theta * self.theta + self.gamma * self.theta - self.beta) \
                    * tf.cast(tf.eye(self.ffn), precision) - self.lambdaM * M_block_1
            else:
                B = (self.alpha * self.theta * self.theta + self.gamma * self.theta - self.beta) \
                    * tf.cast(tf.eye(self.ffn), precision)

            # A
            A = 2.0 * self.theta * self.alpha

            # F
            g = 1.0 / self.g_scale
            Sg = tf.expand_dims(tf.reduce_sum(feature_maps * g, 0), 0)
            Ss = feature_maps * feature_maps * g
            F_ge = feature_maps * (Sg * g) - tf.matmul(feature_maps, Sg * g, transpose_b=True) * feature_maps
            F_ce = -Ss + feature_maps * tf.expand_dims(tf.reduce_sum(Ss, 1), 1)
            F = tf.matmul(frame_patches, self.lambdaE * F_ge + self.lambdaC * F_ce, transpose_a=True)

            # update terms
            D_q1 = tf.matmul(D, q1)
            C_q2 = tf.matmul(C, q2)
            B_q3 = tf.matmul(B, q3)
            A_q4 = A * q4

            gradient_like1 = -q2
            gradient_like2 = -q3
            gradient_like3 = -q4
            gradient_like4 = (D_q1 + C_q2 + F + B_q3 + A_q4) / self.alpha

            # step sizes
            with tf.control_dependencies([gradient_like1, gradient_like2, gradient_like3, gradient_like4]):
                if not self.grad:
                    step_size_up = tf.assign(step_size, tf.ones_like(step_size) * (1.0 - night) * self.step_size)
                else:
                    if self.step_adapt:
                        increase = tf.cast(tf.greater(gradient_like4 * gradient_like_0, 0.0), precision)
                        reduce = 1.0 - increase
                        step_size_up = tf.assign(step_size,
                                                 tf.minimum(tf.maximum(step_size * 0.1 * reduce +
                                                                       step_size * 2.0 * increase,
                                                                       self.step_size), self.step_size * 1000))
                    else:
                        step_size_up = step_size

            # update rules
            with tf.control_dependencies([step_size_up]):
                if not self.grad:
                    up_q1 = tf.assign_sub(q1, gradient_like1 * step_size_up)
                    with tf.control_dependencies([up_q1]):
                        up_q2 = tf.assign_sub(q2, gradient_like2 * step_size_up + q2 * night)
                        with tf.control_dependencies([up_q2]):
                            up_q3 = tf.assign_sub(q3, gradient_like3 * step_size_up + q3 * night)
                            with tf.control_dependencies([up_q3]):
                                up_q4 = tf.assign_sub(q4, gradient_like4 * step_size_up + q4 * night)
                else:
                    up_q4 = tf.assign_sub(q1, gradient_like4 * step_size_up)

            # updating cyclic dependencies and stats
            with tf.control_dependencies([gamma_dot, M_block_dot, N_block_dot, step_size_up, minus_ge2]):
                up_frame_0 = tf.assign(frame_0, frame_1)

                if self.lambdaM > 0.0:
                    up_M_block_0 = tf.assign(M_block_0, M_block_1)
                    up_N_block_0 = tf.assign(N_block_0, N_block_1)
                else:
                    up_M_block_0 = tf.eye(1)
                    up_N_block_0 = tf.eye(1)

                if self.step_adapt and self.grad:
                    up_gradient_like_0 = tf.assign(gradient_like_0, gradient_like4)
                else:
                    up_gradient_like_0 = tf.eye(1)

                up_feature_map_stats = tf.assign(feature_map_stats, biased_maps)

                diff_rho = 1.0 - rho
                up_night = tf.assign(is_night, night)
                up_rho = tf.assign_add(rho, self.eta * tf.cast(tf.greater(diff_rho, 0.0), precision)
                                       * diff_rho * (1.0 - night))

            # coordinator
            with tf.control_dependencies([up_q4, up_frame_0, up_M_block_0, up_N_block_0,
                                          up_gradient_like_0, t_update,
                                          up_feature_map_stats, up_rho, up_night, obj_comp_values_updated]):
                fake_op = tf.eye(1)

            # operations to be executed in the data flow graph (filters_matrix: filter_volume x m)
            out_feature_maps = tf.reshape(feature_maps, [self.h, self.w, self.m])  # h x w x m
            out_filters_map = tf.transpose(tf.reshape(q1, [self.f * self.f, self.n, self.m]))  # m x n x f^2

            # summaries
            tf.summary.scalar("ZA_Norm_FTimesAlpha", tf.reduce_sum(tf.square(F)))
            tf.summary.scalar("ZB_Q00", q1[0][0])
            summary_ops = tf.summary.merge_all()

            ops = [out_feature_maps, out_filters_map,
                   obj_values, obj_comp_values_updated,
                   up_night, up_rho,
                   mi_real_full, motion_full_update,
                   summary_ops,
                   fake_op]

            # initialization
            if not self.resume:
                self.sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

        return ops, frame_0_init_op, saver, t_reset_op

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

    # work-in-progress (Runge-Kutta): this is unused right now
    def __gradient_likes(self, mat_a, mat_b, mat_c, mat_d, g, q1, q2, q3, q4, t, precision):
        D_q1 = tf.matmul(mat_d, q1)
        C_q2 = tf.matmul(mat_c, q2)
        B_q3 = tf.matmul(mat_b, q3)
        A_q4 = mat_a * q4

        current_img, current_of = self.stream.get_next(0.0, t=t, sample_only=True)
        frame_patches = self.__extract_patches(tf.expand_dims(
            tf.div(tf.cast(tf.identity(current_img), precision), 255.0), 0))

        feature_maps = tf.nn.softmax(tf.matmul(frame_patches, q1), dim=1)
        Sg = tf.expand_dims(tf.reduce_sum(feature_maps * g, 0), 0)
        Ss = feature_maps * feature_maps * g
        F_ge = feature_maps * (Sg * g) - tf.matmul(feature_maps, Sg * g, transpose_b=True) * feature_maps
        F_ce = -Ss + feature_maps * tf.expand_dims(tf.reduce_sum(Ss, 1), 1)
        F = tf.matmul(frame_patches, self.lambdaE * F_ge + self.lambdaC * F_ce, transpose_a=True)

        gradient_like1 = tf.identity(-q2)
        gradient_like2 = tf.identity(-q3)
        gradient_like3 = tf.identity(-q4)
        gradient_like4 = tf.identity(D_q1 + C_q2 + F + B_q3 + A_q4) / self.alpha

        return gradient_like1, gradient_like2, gradient_like3, gradient_like4
