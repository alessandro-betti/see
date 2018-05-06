import tensorflow as tf
import numpy as np
import os
import shutil


class FeatureExtractor:

    def __init__(self, w, h, options, resume=False,
                 layer=0, session_data=None, input_data=None, motion_data=None,
                 prev_layers_ops=None, prev_layers_summary_ops=None):
        self.__first_frame = True
        self.step = 1
        self.layer = layer

        # saving options (and other useful values) to direct attributes
        self.f = options['f'][layer]  # full edge of the filter (i.e., 3 in case of 3x3 filters)
        self.n = options['n'][layer]  # input channels/features
        self.m = options['m'][layer]  # output features
        self.init_q = options['init_q'][layer]
        self.alpha = options['alpha'][layer]
        self.beta = options['beta'][layer]
        self.theta = options['theta'][layer]
        self.k = options['k'][layer]
        self.gamma = options['gamma'][layer]
        self.lambdaE = options['lambdaE'][layer]
        self.lambdaC = options['lambdaC'][layer]
        self.lambdaM = options['lambdaM'][layer]
        self.eps1 = options['eps1'][layer] * self.m
        self.eps2 = options['eps2'][layer] * self.m
        self.eps3 = options['eps3'][layer] * self.m
        self.c_eps1 = options['c_eps1'][layer] * self.m
        self.c_eps2 = options['c_eps2'][layer] * self.m
        self.c_eps3 = options['c_eps3'][layer] * self.m
        self.c_frames = options['c_frames'][layer]
        self.c_frames_min = options['c_frames_min'][layer]
        self.gew = options['gew'][layer]
        self.eta = options['eta'][layer]
        self.init_fixed = options['init_fixed'][layer]
        self.rho = options['rho'][layer]
        self.day_only = options['day_only'][layer]

        self.step_size = options['step_size']
        self.step_adapt = options['step_adapt']
        self.all_black = options['all_black']
        self.grad = options['grad']
        self.root = options['root']
        self.rk = options['rk']
        self.stream = options['stream']
        self.grad_order2 = options['grad_order2']

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
            self.theta = 0.0  # this should be >> 1
            self.alpha = 1.0  # this should be 0, but I need this value (1.0) due to implementation issues
            self.beta = 0.0
            self.gamma = 0.0  # this should be 1/(\theta^2)
            self.lambdaM = 0.0

        if self.grad_order2:
            if self.theta < 1000.0:
                self.theta = 1000.0 # this should be >> 1
            self.alpha = 0.0
            self.beta = 0.0
            self.gamma = 0.0  # this should be 1/(\theta^2)
            self.lambdaM = 0.0
            options['check_params'] = False

        self.__check_params(skip_some_checks=not options['check_params'])

        # TensorFlow session and graph
        if self.layer == 0:
            self.sess = tf.Session()
            self.process_frame_ops, self.frame_0_init_op, \
                self.saver, self.t_reset_op, self.motion_01, self.logits, self.rho_op, self.summary_ops = \
                self.__build_graph()
            self.process_frame_ops = [self.process_frame_ops]
            self.summary_ops = [self.summary_ops]
        else:
            self.sess = session_data
            self.process_frame_ops, self.frame_0_init_op, \
                self.saver, self.t_reset_op, self.motion_01, self.logits, self.rho_op, self.summary_ops = \
                self.__build_graph(input_data, motion_data)
            self.process_frame_ops = prev_layers_ops + [self.process_frame_ops]
            self.summary_ops = prev_layers_summary_ops + [self.summary_ops]
        self.activate_tensor_board()

        # TensorFlow model, save dir
        self.model_folder = options['root'] + '/model/num_layers_' + str(self.layer)
        self.save_path = self.model_folder + '/model.saved'
        self.save_path_base = options['root'] + '/model/'

        # creating folders
        self.create_model_folders()

    def close(self, close_session=True):
        self.summary_writer.close()
        if close_session:
            self.sess.close()

    def save(self):
        self.saver.save(self.sess, self.save_path)

    def load(self, steps):
        self.saver.restore(self.sess, self.save_path)
        self.step = steps

    def create_model_folders(self):
        if not os.path.exists(self.save_path_base):
            os.makedirs(self.save_path_base)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

    def activate_tensor_board(self):
        if (not self.resume) and os.path.exists(self.root + '/tensor_board/layer' + str(self.layer)):
            shutil.rmtree(self.root + '/tensor_board/layer' + str(self.layer))
        self.summary_writer = tf.summary.FileWriter(self.root + '/tensor_board/layer' + str(self.layer),
                                                    self.sess.graph)

    def add_to_tensor_board(self, summaries):

        # this method (add_to_tensor_board) is supposed to be called after having called "run_step",
        # but since self.step gets incremented when calling "run_step", here we decrease it by 1
        self.summary_writer.add_summary(summaries, self.step - 1)

    def get_rho(self):
        return self.sess.run(self.rho_op)

    # processing the current frame (frame_1 below)
    def run_step(self, frame_1_to_feed, motion_01_to_feed, gaussian_filter, scaling):

        # zero signal
        if self.all_black > 0:
            frame_1_to_feed.fill(0.0)
            motion_01_to_feed.fill(0.0)

        # quantity: 1 / delta, where delta is the ratio for computing derivatives
        if self.__first_frame and self.step == 1:  # layer-steps start from 1
            one_over_delta = 0.0  # so derivatives will be zero
        else:
            one_over_delta = float(self.step_size)

        # getting values that are fed to the session "runner"
        fed_frame_1 = tf.get_default_graph().get_tensor_by_name("fed_frame_1:0")
        fed_motion_01 = tf.get_default_graph().get_tensor_by_name("fed_motion_01:0")
        fed_one_over_delta = tf.get_default_graph().get_tensor_by_name("fed_one_over_delta:0")
        fed_blur_filter = tf.get_default_graph().get_tensor_by_name("fed_blur_filter:0")
        fed_scaling = tf.get_default_graph().get_tensor_by_name("fed_scaling:0")
        feed_dict = {fed_frame_1: frame_1_to_feed,
                     fed_motion_01: motion_01_to_feed,
                     fed_one_over_delta: one_over_delta,
                     fed_blur_filter: gaussian_filter,
                     fed_scaling: scaling}

        # fixing the case of the first frame
        if self.__first_frame:
            if self.step <= 1:
                self.sess.run(self.frame_0_init_op, feed_dict=feed_dict)
            if self.step == 0:  # layer-steps start from 1, so the value "0" is a special code to say "reset t, please!"
                self.sess.run(self.t_reset_op, feed_dict=feed_dict)
                self.step = 1  # we put back the original starting value for layer-steps, that is 1
            self.__first_frame = False

        # running the computations over the TensorFlow graph
        ret, summaries = self.sess.run([self.process_frame_ops, self.summary_ops], feed_dict=feed_dict)

        # next step
        self.step = self.step + 1

        # returning data (no output printing in this class, please!)
        return ret, summaries

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
                raise ValueError("Invalid alpha: " + str(self.alpha) + " (it must be > 0)")

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

    def __build_graph(self, input_data=None, motion_data=None):

        # TensorFlow precision
        precision = tf.float32

        # TensorFlow inputs
        if self.layer == 0:
            assert (input_data is None) and (motion_data is  None), "Invalid configuration!"
        else:
            assert (input_data is not None) and (motion_data is not None), "Invalid configuration!"

        if self.layer == 0:
            fed_frame_1 = tf.placeholder(precision, name="fed_frame_1")
            fed_motion_01 = tf.placeholder(precision, name="fed_motion_01")
            fed_one_over_delta = tf.placeholder(precision, shape=(), name="fed_one_over_delta")
            fed_blur_filter = tf.placeholder(precision, name="fed_blur_filter")
            fed_scaling = tf.placeholder(precision, name="fed_scaling")
        else:
            fed_one_over_delta = tf.get_default_graph().get_tensor_by_name("fed_one_over_delta:0")
            fed_blur_filter = tf.get_default_graph().get_tensor_by_name("fed_blur_filter:0")
            fed_scaling = tf.get_default_graph().get_tensor_by_name("fed_scaling:0")

        # TensorFlow variables (main scope)
        layer_scope = "layer" + str(self.layer)
        with tf.variable_scope(layer_scope, reuse=False):
            rho = tf.get_variable("rho", (), dtype=precision,
                                  initializer=tf.constant_initializer(self.rho, dtype=precision))
            is_night = tf.get_variable("night", (), dtype=precision,
                                       initializer=tf.constant_initializer(0.0, dtype=precision))
            t = tf.get_variable("t", (), dtype=precision,
                                initializer=tf.constant_initializer(0.0, dtype=precision))
            avg_p_full = tf.get_variable("avg_p_full", [self.m], dtype=precision,
                                         initializer=tf.constant_initializer(0.0, dtype=precision))
            avg_p_log_p_full = tf.get_variable("avg_p_log_p_full", [self.m], dtype=precision,
                                               initializer=tf.constant_initializer(0.0, dtype=precision))
            motion_full = tf.get_variable("motion_full", [2], dtype=precision,
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
            winning_features_0 = tf.get_variable("winning_features_0", [self.wh], dtype=tf.int32,
                                                 initializer=tf.zeros_initializer)

            if self.lambdaM > 0.0:
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
            if self.layer == 0:
                input_scale = 255.0
                frame_0_init_op = tf.assign(frame_0, tf.expand_dims(tf.div(fed_frame_1, input_scale), 0))
                frame_1 = tf.expand_dims(tf.div(fed_frame_1, input_scale), 0)  # adding fake batch dim 1 x h x w x n
            else:
                input_scale = 1.0
                frame_0_init_op = tf.assign(frame_0, tf.expand_dims(tf.div(input_data, input_scale), 0))
                frame_1 = tf.expand_dims(tf.div(input_data, input_scale), 0)  # adding fake batch dim 1 x h x w x n
                frame_1 = self.__blur(frame_1, fed_blur_filter, fed_scaling)  # blurring/scaling

            t_reset_op = [tf.assign(t, 0.0), tf.assign(rho, self.rho)]

            if self.layer == 0:
                motion_01 = tf.expand_dims(fed_motion_01, 3)  # h x w x 2 x 1 (the 1st motion comp. is horizontal)
            else:
                motion_01 = motion_data

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

            # convolution
            logits_maps = tf.matmul(frame_patches, q1)
            feature_maps = tf.nn.softmax(logits_maps, axis=1)  # wh x m

            # out_logits_maps = tf.reshape(logits_maps, [self.h, self.w, self.m])
            out_logits_maps = tf.reshape(tf.div(feature_maps, tf.reduce_max(feature_maps, 1, keepdims=True)),
                                         [self.h, self.w, self.m])

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

                # getting the spatial gradient (h x w x 2 x n (the first spatial component is horizontal))
                spatial_gradient = FeatureExtractor.__spatial_gradient(frame_1, self.h, self.w, self.n)  # frame 1 here

                # mixing the spatial gradient with motion (element-wise product + sum): h x w x n
                v_delta_gamma = tf.reduce_sum(tf.multiply(spatial_gradient,
                                                          motion_01), 2)  # broadcast over "n", then sum on x,y
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

                # obj term
                motion = tf.reduce_sum(tf.multiply(q2, tf.matmul(M_block_1, q2))) \
                    + 2.0 * tf.reduce_sum(tf.multiply(q1, tf.matmul(N_block_1, q2))) \
                    + tf.reduce_sum(tf.multiply(q1, tf.matmul(O_block, q1)))

                # measure term
                winning_features_1 = tf.argmax(feature_maps, axis=1, output_type=tf.int32)
                i = tf.tile(tf.reshape(tf.cast(tf.range(self.h), tf.float32), [self.h, 1]), [1, self.w])
                j = tf.tile(tf.reshape(tf.cast(tf.range(self.w), tf.float32), [1, self.w]), [self.h, 1])
                ii = tf.minimum(tf.maximum(tf.cast(tf.round(i - motion_01[:,:,1,0]), tf.int32), 0), self.h-1)
                jj = tf.minimum(tf.maximum(tf.cast(tf.round(j - motion_01[:,:,0,0]), tf.int32), 0), self.w-1)
                ii_jj = tf.reshape(ii * self.w + jj, [self.wh])
                target_winning_features_0 = tf.gather(winning_features_0, ii_jj)
                is_first_frame = tf.cast(tf.minimum(t, 1.0), tf.int32)
                motion_acc = 1.0 - tf.div(tf.count_nonzero(is_first_frame * winning_features_1 -
                                                           target_winning_features_0, dtype=tf.float32), float(self.wh))

            else:

                # obj term
                motion = 0.0

                # measure term
                is_first_frame = tf.eye(1)
                winning_features_1 = tf.zeros_like(winning_features_0)
                motion_acc = 0.0

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

            # scaling
            w = tf.exp(self.theta * t)
            scaling = obj_comp_values[0] + w

            # updating stats
            with tf.control_dependencies([scaling,is_first_frame]):
                up_t = tf.assign_add(t, 1.0)
                avg_p_full_update = tf.assign_add(avg_p_full, (avg_p - avg_p_full) / up_t)
                avg_p_log_p_full_update = tf.assign_add(avg_p_log_p_full, (avg_p_log_p - avg_p_log_p_full) / up_t)
                ce_real_full = -tf.reduce_sum(avg_p_log_p_full_update)  # this is already in [0,1]
                minus_ge_real_full2 = tf.reduce_sum(
                    tf.multiply(avg_p_full_update,
                                tf.div(tf.log(avg_p_full_update), np.log(self.m))))  # this is in [-1,0]
                mi_real_full = (-ce_real_full) - minus_ge_real_full2
                minus_ge_real_full = minus_ge_real_full2 + 1.0  # translating into [0,1]
                motion_full_update = tf.assign_add(motion_full, (tf.stack([motion, motion_acc]) - motion_full) / up_t)

            # average on objective function terms
            obj_comp = (obj_comp_values[1] * obj_comp_values[0] + obj * w) / scaling
            ce_comp = (obj_comp_values[2] * obj_comp_values[0] + ce * w) / scaling
            minus_ge_comp = (obj_comp_values[3] * obj_comp_values[0] + minus_ge * w) / scaling
            mi_comp = (obj_comp_values[4] * obj_comp_values[0] + mi * w) / scaling
            mi_real_comp = (obj_comp_values[5] * obj_comp_values[0] + mi_real * w) / scaling
            motion_comp = (obj_comp_values[6] * obj_comp_values[0] + motion * w) / scaling
            norm_q_comp = (obj_comp_values[7] * obj_comp_values[0] + norm_q * w) / scaling
            norm_q_dot_comp = (obj_comp_values[8] * obj_comp_values[0] + norm_q_dot * w) / scaling
            norm_q_dot_dot_comp = (obj_comp_values[9] * obj_comp_values[0] + norm_q_dot_dot * w) / scaling
            norm_q_mixed_comp = (obj_comp_values[10] * obj_comp_values[0] + norm_q_mixed * w) / scaling
            norm_q_dot_dot_dot_comp = (obj_comp_values[11] * obj_comp_values[0] + norm_q_dot_dot_dot * w) / scaling

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
            tf.summary.scalar("AD_RealConditionalEntropyFull", ce_real_full)
            tf.summary.scalar("AE_RealMinusEntropyFull", minus_ge_real_full)
            tf.summary.scalar("AF_MotionFull", motion_full_update[0])
            tf.summary.scalar("AG_MotionAccFull", motion_full_update[1])

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

            # updating cyclic dependencies and stats (forward-related)
            with tf.control_dependencies([obj_comp_values_updated]):
                up_frame_0 = tf.assign(frame_0, frame_1)
                up_feature_map_stats = tf.assign(feature_map_stats, biased_maps)
                up_winning_features_0 = tf.assign(winning_features_0, winning_features_1)
                diff_rho = 1.0 - rho
                up_night = tf.assign(is_night, night)
                up_rho = tf.assign_add(rho, self.eta * tf.cast(tf.greater(diff_rho, 0.0), precision)
                                       * diff_rho * (1.0 - night))

            # forward operations
            with tf.control_dependencies([obj_comp_values_updated,
                                          up_t, up_frame_0, up_winning_features_0,
                                          up_feature_map_stats, up_rho, up_night]):
                forward = tf.eye(1)

            # intermediate terms
            if self.lambdaM > 0.0:
                N_block_1_trans = tf.transpose(N_block_1)  # filter_volume x filter_volume

                # other derivatives over time (fed_one_over_delta = 0 when t = 0)
                M_block_dot = tf.multiply(tf.subtract(M_block_1, M_block_0),
                                          fed_one_over_delta)  # filter vol x filter vol
                N_block_dot = tf.multiply(tf.subtract(N_block_1, N_block_0),
                                          fed_one_over_delta)  # filter vol x filter vol
            else:
                M_block_dot = tf.eye(1)
                N_block_dot = tf.eye(1)

            # D (this is just a portion of the D matrix in the paper) - q^(3): q1 in the code
            if self.lambdaM > 0.0:
                D = self.k * tf.cast(tf.eye(self.ffn), precision) \
                    - (self.lambdaM * self.theta) * N_block_1_trans \
                    + self.lambdaM * tf.subtract(O_block, tf.transpose(N_block_dot))
            else:
                if not self.grad_order2:
                    D = self.k * tf.cast(tf.eye(self.ffn), precision)

            # C - q^(2): q2 in the code
            if self.lambdaM > 0.0:
                C = (self.gamma * self.theta * self.theta - self.beta * self.theta) * \
                    tf.cast(tf.eye(self.ffn), precision) \
                    - (self.lambdaM * self.theta) * M_block_1 \
                    - self.lambdaM * (M_block_dot + N_block_1_trans - N_block_1)
            else:
                if not self.grad_order2:
                    C = (self.gamma * self.theta * self.theta - self.beta * self.theta) * \
                        tf.cast(tf.eye(self.ffn), precision)

            # B - q^(1): q3 in the code
            if self.lambdaM > 0.0:
                B = (self.alpha * self.theta * self.theta + self.gamma * self.theta - self.beta) \
                    * tf.cast(tf.eye(self.ffn), precision) - self.lambdaM * M_block_1
            else:
                if not self.grad_order2:
                    B = (self.alpha * self.theta * self.theta + self.gamma * self.theta - self.beta) \
                        * tf.cast(tf.eye(self.ffn), precision)

            # A - q: q4 in the code
            if not self.grad_order2:
                A = 2.0 * self.theta * self.alpha

            # F - nothing
            g = 1.0 / self.g_scale
            Sg = tf.expand_dims(tf.reduce_sum(feature_maps * g, 0), 0)
            Ss = feature_maps * feature_maps * g
            F_ge = feature_maps * (Sg * g) - tf.matmul(feature_maps, Sg * g, transpose_b=True) * feature_maps
            F_ce = -Ss + feature_maps * tf.expand_dims(tf.reduce_sum(Ss, 1), 1)
            F = tf.matmul(frame_patches, self.lambdaE * F_ge + self.lambdaC * F_ce, transpose_a=True)

            # update terms
            if not self.grad_order2:
                D_q1 = tf.matmul(D, q1)
                C_q2 = tf.matmul(C, q2)
                B_q3 = tf.matmul(B, q3)
                A_q4 = A * q4

                gradient_like1 = -q2
                gradient_like2 = -q3
                gradient_like3 = -q4
                gradient_like4 = (D_q1 + C_q2 + F + B_q3 + A_q4) / self.alpha
            else:
                B = (1.0 / self.theta) * tf.cast(tf.eye(self.ffn), precision)
                D = self.k * tf.cast(tf.eye(self.ffn), precision)

                D_q1 = tf.matmul(D, q1)
                B_q2 = tf.matmul(B, q2)

                gradient_like1 = -q2
                gradient_like2 = D_q1 + F + B_q2
                gradient_like3 = tf.zeros_like(q1)
                gradient_like4 = tf.zeros_like(q1)

            # step sizes
            with tf.control_dependencies([gradient_like1, gradient_like2, gradient_like3, gradient_like4]):
                if not self.grad:
                    step_size_up = tf.assign(step_size, tf.ones_like(step_size) * (1.0 - night) * self.step_size)
                else:
                    if self.step_adapt:
                        increase = tf.cast(tf.greater(gradient_like4 * gradient_like_0, 0.0), precision)
                        reduce = 1.0 - increase
                        step_size_up = tf.assign(step_size,
                                                 tf.minimum(tf.maximum(step_size * 0.1 * reduce + step_size * 2.0 * increase,
                                                            self.step_size), self.step_size * 1000))
                    else:
                        step_size_up = step_size

            # updating cyclic dependencies and stats (backward-related)
            with tf.control_dependencies([step_size_up]):
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

            # update rules
            with tf.control_dependencies([step_size_up]):
                if not self.grad:
                    if not self.grad_order2:
                        up_q1 = tf.assign_sub(q1, gradient_like1 * step_size_up)
                        with tf.control_dependencies([up_q1]):
                            up_q2 = tf.assign_sub(q2, gradient_like2 * step_size_up + q2 * night)
                            with tf.control_dependencies([up_q2]):
                                up_q3 = tf.assign_sub(q3, gradient_like3 * step_size_up + q3 * night)
                                with tf.control_dependencies([up_q3]):
                                    up_q4 = tf.assign_sub(q4, gradient_like4 * step_size_up + q4 * night)
                    else:
                        up_q1 = tf.assign_sub(q1, gradient_like1 * step_size_up)
                        with tf.control_dependencies([up_q1]):
                            up_q4 = tf.assign_sub(q2, gradient_like2 * step_size_up + q2 * night)
                else:
                    up_q4 = tf.assign_sub(q1, gradient_like4 * step_size_up)

            # backward and update operations
            with tf.control_dependencies([up_q4, up_M_block_0, up_N_block_0, up_gradient_like_0]):
                backward_and_update = tf.eye(1)

            # operations to be executed in the data flow graph (filters_matrix: filter_volume x m)
            out_feature_maps = tf.reshape(feature_maps, [self.h, self.w, self.m])  # h x w x m
            out_filters_map = tf.transpose(tf.reshape(q1, [self.f * self.f, self.n, self.m]))  # m x n x f^2

            # summaries
            tf.summary.scalar("ZA_Norm_FTimesAlpha", tf.reduce_sum(tf.square(F)))
            tf.summary.scalar("ZB_Q00", q1[0][0])
            summary_ops = tf.summary.merge_all(scope=layer_scope)

            ops = [out_feature_maps, out_filters_map,
                   obj_values, obj_comp_values_updated,
                   [up_night, up_rho],
                   [mi_real_full, ce_real_full, minus_ge_real_full, motion_full_update[0], motion_full_update[1]],
                   forward, backward_and_update]

            # initialization
            if not self.resume:
                self.sess.run(tf.variables_initializer(tf.global_variables(layer_scope)))  # layer-wise initialization

            saver = tf.train.Saver()

        return ops, frame_0_init_op, saver, t_reset_op, motion_01, out_logits_maps, rho, summary_ops

    @staticmethod
    def __spatial_gradient(source, h, w, n):
        channel_filter = tf.eye(n)
        filter_pattern = tf.reshape(tf.constant([-1.0, 0.0, +1.0], source.dtype), [1,3])

        left_right_filter = (tf.expand_dims(tf.expand_dims(filter_pattern, -1), -1) *
                             tf.expand_dims(tf.expand_dims(channel_filter, 0), 0))
        up_down_filter = (tf.expand_dims(tf.expand_dims(tf.transpose(filter_pattern), -1), -1) *
                          tf.expand_dims(tf.expand_dims(channel_filter, 0), 0))

        # 1 x h x w x n (each)
        left_right_grad = tf.nn.conv2d(source, left_right_filter, strides=[1, 1, 1, 1], padding='SAME')
        up_down_grad = tf.nn.conv2d(source, up_down_filter, strides=[1, 1, 1, 1], padding='SAME')

        # returns: h x w x 2 x n (the first spatial component - 3rd axis - is horizontal)
        return tf.reshape(tf.stack([left_right_grad, up_down_grad], axis=3), [h, w, 2, n])

        # up_down_filter = tf.reshape(tf.constant([-1, 0, +1], source.dtype), [3, 1, 1, 1])
        # left_right_filter = tf.reshape(up_down_filter, [1, 3, 1, 1])
        #
        # input_t = tf.transpose(source)  # n x w x h x 1
        #
        # # n x w x h x 1 (each)
        # up_down_grad = tf.nn.conv2d(input_t, up_down_filter, strides=[1, 1, 1, 1], padding='SAME')
        # left_right_grad = tf.nn.conv2d(input_t, left_right_filter, strides=[1, 1, 1, 1], padding='SAME')
        #
        # # returns: h x w x 2 x n (the first spatial component - 3rd axis - is horizontal)
        # return tf.reshape(tf.transpose(tf.stack([up_down_grad, left_right_grad], axis=1)), [h, w, 2, n])

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

        feature_maps = tf.nn.softmax(tf.matmul(frame_patches, q1), axis=1)
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

    def __blur(self, image, pixel_filter, scaling):
        pixel_filter = tf.reshape(pixel_filter, [tf.size(pixel_filter),1])
        channel_filter = tf.eye(self.n)
        _filter1 = (tf.expand_dims(tf.expand_dims(pixel_filter, -1), -1) *
                    tf.expand_dims(tf.expand_dims(channel_filter, 0), 0))
        _filter2 = (tf.expand_dims(tf.expand_dims(tf.transpose(pixel_filter), -1), -1) *
                    tf.expand_dims(tf.expand_dims(channel_filter, 0), 0))
        result_batch1 = tf.nn.conv2d(image,
                                     filter=_filter1,
                                     strides=[1, 1, 1, 1],
                                     padding='SAME')
        result_batch2 = tf.nn.conv2d(result_batch1,
                                     filter=_filter2,
                                     strides=[1, 1, 1, 1],
                                     padding='SAME')
        return scaling * result_batch2
