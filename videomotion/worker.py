import time
from feature_extractor import FeatureExtractor
from utils import out
import json
import os
import numpy as np


class Worker:

    def __init__(self, input_stream, output_stream, w=-1, h=-1, fps=-1, frames=-1,
                 force_gray=False, repetitions=1, options=None, resume=False, reset_stream_when_resuming=False):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.repetitions = repetitions
        self.__completed_repetitions = 0
        self.__start_time = None
        self.__elapsed_time = None
        self.__layer = 0
        self.__layers = options['layers']
        self.steps = 0.0
        self.measured_fps = 0.0
        self.save_scores_only = options['save_scores_only']
        options['stream'] = self.input_stream
        self.input_stream.set_options(w, h, fps, force_gray, frames)
        self.options = options
        self.w = w
        self.h = h
        self.layer_steps = 0

        self.fe = []
        self.log = []
        self.log.append([])

        self.__rho = []
        for i in range(0, self.__layers):
            self.__rho.append(options['rho'][i])

        if resume:
            out("RESUMING...")
            self.fe.append(FeatureExtractor(w, h, options, True))  # here is the TensorFlow based feature extractor!
            self.load(reset_stream_when_resuming)
        else:
            self.fe.append(FeatureExtractor(w, h, options, False))  # here is the TensorFlow based feature extractor!

    def close(self):
        for i in range(0, self.__layer):
            self.fe[i].close(close_session=False)
        self.fe[self.__layer].close(close_session=True)

    def save(self):
        self.fe[self.__layer].save()

        info = {'last_layer': self.__layer,
                'last_layer_steps': self.layer_steps,
                'steps': self.steps,
                'completed_repetitions': self.__completed_repetitions,
                'frame': self.input_stream.get_last_frame_number(),
                'time': self.input_stream.get_last_frame_time(),
                'last_saved_frame': self.output_stream.get_last_frame()}

        f = open(self.fe[0].save_path_base + "/info.txt", "w")
        if f is None or not f or f.closed:
            raise IOError("Cannot access: " + self.fe[0].save_path_base + "/info.txt")
        json.dump(info, f, indent=4)
        f.close()

        for i in range(0, self.__layer + 1):
            layer_log = self.fe[0].save_path_base + "/log_layer" + str(i) + ".txt"

            if not os.path.isfile(layer_log):
                captions = ['steps'] + \
                           ['unused', 'obj', 'ce', 'minus_ge', 'mi', 'mi_real', 'motion', 'norm_q', 'norm_q_dot',
                            'norm_q_dot_dot', 'norm_q_mixed', 'norm_q_dot_dot_dot'] + \
                           ['scaling', 'obj_comp', 'ce', 'minus_ge_comp', 'mi_comp', 'mi_real_comp',
                            'motion_comp', 'norm_q_comp', 'norm_q_dot_comp', 'norm_q_dot_dot_comp',
                            'norm_q_mixed_comp', 'norm_q_dot_dot_dot_comp'] + \
                           ['is_night', 'rho'] + \
                           ['mi_real_full', 'ce_real_full', 'minus_ge_real_full', 'motion_full', 'motion_acc_full']

                f = open(layer_log, "w")
                if f is None or not f or f.closed:
                    raise IOError("Cannot access: " + layer_log)
                for j in range(0, len(captions)-1):
                    f.write("%s" % captions[j])
                    f.write("(" + str(j) + "),")
                f.write("%s" % captions[len(captions)-1])
                f.write("(" + str(len(captions)-1) + ")\n")
                f.close()

            f = open(layer_log, "a")
            if f is None or not f or f.closed:
                raise IOError("Cannot access: " + layer_log)
            for r in range(0, len(self.log[i])):
                for j in range(0, len(self.log[i][r]) - 1):
                    f.write("%s," % self.log[i][r][j])
                f.write("%s\n" % self.log[i][r][len(self.log[i][r]) - 1])
            f.close()

            self.log[i] = []  # clearing

    def load(self, reset_stream=False):
        f = open(self.fe[0].save_path_base + "/info.txt", "r")
        if f is None or not f or f.closed:
            raise IOError("Cannot access: " + self.fe[0].save_path_base + "/info.txt")
        info = json.load(f)
        f.close()

        self.__layer = info['last_layer']

        if self.__layer >= 1:
            self.fe[0].process_frame_ops[-1].pop()  # freezing: the last operation is "backward and update"

        for i in range(1, self.__layer + 1):
            prev_layer_session = self.fe[i - 1].sess
            # prev_layer_output = self.fe[i - 1].process_frame_ops[i - 1][0]
            prev_layer_output = self.fe[i - 1].logits
            prev_layer_motion = self.fe[i - 1].motion_01
            prev_layers_ops = self.fe[i - 1].process_frame_ops
            prev_layers_summary_ops = self.fe[i - 1].summary_ops
            self.fe.append(FeatureExtractor(self.w, self.h, self.options, True,
                                            i,
                                            prev_layer_session,
                                            prev_layer_output,
                                            prev_layer_motion,
                                            prev_layers_ops,
                                            prev_layers_summary_ops))
            if i < self.__layer:
                self.fe[i].process_frame_ops[-1].pop()  # freezing: the last operation is "backward and update"

        if not reset_stream:

            # resume stream
            self.input_stream.get_next(sample_only=True)  # ensure that the stream is open
            self.input_stream.set_last_frame_and_time(info['frame'], info['time'])
            self.output_stream.set_last_frame(info['last_saved_frame'])
            self.__completed_repetitions = info['completed_repetitions']

            # worker-steps
            self.steps = info['steps']  # worker-steps start from 0
            self.layer_steps = info['last_layer_steps']

            # layer-steps
            for i in range(0, self.__layer):
                self.fe[i].step = self.steps + 1  # layer-steps start from 1

            # creating log-lists (one list was already created in the constructor)
            for i in range(0, self.__layer):
                self.log.append([])

            # load (and set number of steps)
            self.fe[self.__layer].load(self.steps + 1)  # layer-steps start from 1

            # get rho
            for i in range(0, self.__layer + 1):
                self.__rho[i] = self.fe[i].get_rho()
        else:

            # do not resume stream
            self.__completed_repetitions = 0

            # worker-steps
            self.steps = 0.0  # worker-steps start from 0
            self.layer_steps = 0  # worker-steps start from 0

            # layer-steps
            for i in range(0, self.__layer):
                self.fe[i].step = 1  # layer-steps start from 1

            # creating log-lists (one list was already created in the constructor)
            for i in range(0, self.__layer):
                self.log.append([])

            # load
            self.fe[self.__layer].load(0)  # this "0" is a special code to say "reset t!" (steps will then be set to 1)

            # get rho
            for i in range(0, self.__layer + 1):
                self.__rho[i] = self.fe[i].get_rho()

            # prepare initialization/reset operations
            frame_0_init_ops = []
            t_reset_ops = []
            for i in range(0, self.__layer + 1):
                frame_0_init_ops.append(self.fe[i].frame_0_init_op)
                t_reset_ops.append(self.fe[i].t_reset_op)
            self.fe[self.__layer].frame_0_init_op = frame_0_init_ops
            self.fe[self.__layer].t_reset_ops = t_reset_ops

            # clearing output, model, tensor-board folders and recreating them
            self.output_stream.create_folders(True)

            # recreating tensor-board folders (they were deleted by the "clearing" operation above)
            for i in range(0, self.__layer + 1):
                self.fe[i].activate_tensor_board()

            # recreating the folder where the model is saved (it was deleted by the "clearing" operation above)
            self.fe[self.__layer].create_model_folders()

            # recreating the option file
            self.output_stream.save_option(opt=None)

    def run_step(self):

        while True:

            # get time
            status = True
            step_time = time.time()
            if self.__start_time is None:
                self.__start_time = step_time

            # get the frame to process at the next step and the currently needed motion field
            step_load_time = time.time()
            current_img, current_of, gaussian_filter, scaling = \
                self.input_stream.get_next(blur_factor=(1.0-self.__rho[0]))

            # handling repetitions
            if current_img is None:
                self.__completed_repetitions = self.__completed_repetitions + 1
                if self.__completed_repetitions < self.repetitions:
                    self.input_stream.reset()
                else:
                    break
            else:
                break

        # check if the stream has ended
        if current_img is not None:
            step_load_time = time.time() - step_load_time

            # extracting features
            outcomes_by_layer, summaries_by_layer = self.fe[self.__layer].run_step(current_img, current_of,
                                                                                   gaussian_filter, scaling)
            features_by_layer = []
            filters_by_layer = []
            others_by_layer = []
            self.layer_steps = self.layer_steps + 1

            for i in range(0, self.__layer + 1):
                self.fe[i].add_to_tensor_board(summaries_by_layer[i])

                if i == self.__layer:
                    features, filters, obj_values, obj_comp_values, is_night_next_rho, full_values, \
                        unused_forward, unused_backward = outcomes_by_layer[i]
                else:
                    self.fe[i].step = self.fe[i].step + 1  # the frozen layers need to be manually handled...
                    features, filters, obj_values, obj_comp_values, is_night_next_rho, full_values, \
                        unused_forward = outcomes_by_layer[i]

                # output-info
                if is_night_next_rho[0] == 1.0:
                    light = "night"
                else:
                    light = "day"

                out("\t[layer=" + str(i) + ", status=" + light + ", rho=" + str(self.__rho[i])
                    + ", action_cur=" + str(obj_values[1])
                    + ", mi_real_full=" + str(full_values[0]) + ", motion_full=" + str(full_values[3])
                    + ", motion_acc_full=" + str(full_values[4])
                    + ",\n\t mi_real=" + str(obj_values[5]) + ", mi=" + str(obj_values[4])
                    + ", ce=" + str(obj_values[2])
                    + ", minus_ge=" + str(obj_values[3])
                    + ", motion=" + str(obj_values[6])
                    + ",\n\t norm_q=" + str(obj_values[7]) + ", q'q''=" + str(obj_values[10])
                    + ", norm_q'=" + str(obj_values[8]) + "/" + "{0:.2f}".format(self.fe[i].eps1)
                    + ", norm_q''=" + str(obj_values[9]) + "/" + "{0:.2f}".format(self.fe[i].eps2)
                    + ", norm_q'''=" + str(obj_values[11]) + "/" + "{0:.2f}".format(self.fe[i].eps3) + "]")

                others = {'layer': int(i), 'last_layer': self.__layer, 'status': light, 'rho': float(self.__rho[i]),
                          'action_cur': float(obj_values[1]),
                          'mi_real_full': float(full_values[0]), 'ce_real_full': float(full_values[1]),
                          'minus_ge_real_full': float(full_values[2]), 'motion_full': float(full_values[3]),
                          'motion_acc_full': float(full_values[4]),
                          'mi_real': float(obj_values[5]), 'mi': float(obj_values[4]),
                          'ce': float(obj_values[2]),
                          'minus_ge': float(obj_values[3]),
                          'motion': float(obj_values[6]), 'norm_q': float(obj_values[7]),
                          'norm_q_mixed': float(obj_values[10]),
                          'norm_q_dot': float(obj_values[8]),
                          'norm_q_dot_dot': float(obj_values[9]),
                          'norm_q_dot_dot_dot': float(obj_values[11]),
                          'eps1': self.fe[i].eps1,
                          'eps2': self.fe[i].eps2,
                          'eps3': self.fe[i].eps3}

                self.log[i].append(np.append(np.append(np.append(np.append([self.steps + 1],  # layer-steps start from 1
                                                                           obj_values),
                                                                 obj_comp_values),
                                                       is_night_next_rho),
                                             full_values))

                features_by_layer.append(features)
                filters_by_layer.append(filters)
                others_by_layer.append(others)

                # updating rho (print only)
                self.__rho[i] = is_night_next_rho[1]

                # print("FEATURES:")
                # print(features)

                # print("FILTERS:")
                # print(filters)

                # checking errors
                # if not np.isfinite(filters).any() or np.isnan(filters).any():
                #    raise ValueError("Filters contain NaNs or Infinite!")
                # if not np.isfinite(features).any() or np.isnan(features).any():
                #    raise ValueError("Feature maps contain NaNs or Infinite!")

            # save output
            step_save_time = time.time()
            if not self.save_scores_only:
                self.output_stream.save_next(current_img, current_of,
                                             features_by_layer, filters_by_layer, others_by_layer)

            step_save_time = time.time() - step_save_time

            # stats
            step_time = time.time() - step_time
            self.__elapsed_time = time.time() - self.__start_time
            self.steps = self.steps + 1.0
            self.measured_fps = self.steps / self.__elapsed_time

            # saving model (every 1000 steps)
            if int(self.steps) % 1000 == 0:
                self.save()

            # next layer activation
            norm_q_dot = obj_values[8]
            norm_q_dot_dot = obj_values[9]
            norm_q_dot_dot_dot = obj_values[11]

            if self.layer_steps >= self.fe[self.__layer].c_frames_min and \
                    (self.layer_steps >= self.fe[self.__layer].c_frames or
                    (norm_q_dot < self.fe[self.__layer].c_eps1
                     and norm_q_dot_dot < self.fe[self.__layer].c_eps2
                     and norm_q_dot_dot_dot < self.fe[self.__layer].c_eps3)):

                if (self.__layer + 1) < self.__layers:
                    out("Activating a new layer...")
                    self.layer_steps = 0
                    self.save()
                    self.__layer = self.__layer + 1
                    self.log.append([])

                    prev_layer_session = self.fe[self.__layer - 1].sess
                    # prev_layer_output = self.fe[self.__layer - 1].process_frame_ops[self.__layer - 1][0]
                    prev_layer_output = self.fe[self.__layer - 1].logits
                    prev_layer_motion = self.fe[self.__layer - 1].motion_01
                    prev_layers_ops = self.fe[self.__layer - 1].process_frame_ops
                    prev_layers_ops[-1].pop()  # freezing: the last operation is "backward and update"
                    prev_layers_summary_ops = self.fe[self.__layer - 1].summary_ops
                    self.fe.append(FeatureExtractor(self.w, self.h, self.options, False,
                                                    self.__layer,
                                                    prev_layer_session,
                                                    prev_layer_output,
                                                    prev_layer_motion,
                                                    prev_layers_ops,
                                                    prev_layers_summary_ops))
                    self.fe[self.__layer].step = self.steps + 1  # layer-steps start from 1

        else:
            status = False
            step_load_time = time.time() - step_load_time
            step_save_time = 0.0

            # stats
            step_time = time.time() - step_time
            self.__elapsed_time = time.time() - self.__start_time

        return status, step_time, (step_time - step_save_time), (step_time - step_save_time - step_load_time)
