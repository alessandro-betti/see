import numpy as np
import time
from feature_extractor import FeatureExtractor
from utils import out
import json


class Worker:

    def __init__(self, input_stream, output_stream, w=-1, h=-1, fps=-1, frames=-1,
                 force_gray=False, repetitions=1, options=None, resume=False, reset_stream_when_resuming=False):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.repetitions = repetitions
        self.__completed_repetitions = 0
        self.__start_time = None
        self.__elapsed_time = None
        self.__rho = options['rho']
        self.steps = 0.0
        self.measured_fps = 0.0
        self.save_scores_only = options['save_scores_only']
        options['stream'] = self.input_stream
        self.input_stream.set_options(w, h, fps, force_gray, frames)
        self.fe = FeatureExtractor(w, h, options, resume)  # here is the TensorFlow based feature extractor!
        self.blink_steps = []

        if resume:
            out("RESUMING...")
            self.load(reset_stream_when_resuming)

    def close(self):
        self.fe.close()

    def save(self):
        self.fe.save()

        info = {'steps': self.steps,
                'frame': self.input_stream.get_last_frame_number(),
                'time': self.input_stream.get_last_frame_time(),
                'blink_steps': self.blink_steps}

        f = open(self.fe.save_path + ".info.txt", "w")
        if f is None or not f or f.closed:
            raise IOError("Cannot access: " + self.fe.save_path + ".info.txt")
        json.dump(info, f, indent=4)
        f.close()

    def load(self, reset_stream=False):
        if not reset_stream:
            f = open(self.fe.save_path + ".info.txt", "r")
            if f is None or not f or f.closed:
                raise IOError("Cannot access: " + self.fe.save_path + ".info.txt")
            info = json.load(f)
            f.close()

            self.steps = info['steps']
            self.fe.load(self.steps)

            self.input_stream.get_next(sample_only=True)  # ensure that the stream is open
            self.input_stream.set_last_frame_and_time(info['frame'], info['time'])
            self.output_stream.set_last_frame(info['frame'])
        else:
            self.steps = 0.0
            self.fe.load(1)
            self.output_stream.create_folders(True)  # clearing folders and recreating them
            self.fe.activate_tensor_board()

    def run_step(self):

        while True:

            # get time
            status = True
            step_time = time.time()
            if self.__start_time is None:
                self.__start_time = step_time

            # get the frame to process at the next step and the currently needed motion field
            step_load_time = time.time()
            current_img, current_of = self.input_stream.get_next(blur_factor=(1.0-self.__rho))

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
            features, filters, obj_values, obj_comp_values, is_night, next_rho, mi_real_full, motion_full = \
                self.fe.run_step(current_img, current_of)

            # output-info
            if is_night == 1.0:
                light = "night"
            else:
                light = "day"

            out("\t[status=" + light + ", rho=" + str(self.__rho) + ", action_cur=" + str(obj_values[1])
                + ", mi_real_full=" + str(mi_real_full) + ", motion_full=" + str(motion_full)
                + ",\n\t mi_real=" + str(obj_values[5]) + ", mi=" + str(obj_values[4])
                + ", ce=" + str(obj_values[2])
                + ", minus_ge=" + str(obj_values[3])
                + ", motion=" + str(obj_values[6])
                + ",\n\t norm_q=" + str(obj_values[7]) + ", q'q''=" + str(obj_values[10])
                + ", norm_q'=" + str(obj_values[8]) + "/" + "{0:.2f}".format(self.fe.eps1)
                + ", norm_q''=" + str(obj_values[9]) + "/" + "{0:.2f}".format(self.fe.eps2)
                + ", norm_q'''=" + str(obj_values[11]) + "/" + "{0:.2f}".format(self.fe.eps3) + "]")

            others = {'status': light, 'rho': float(self.__rho), 'action_cur': float(obj_values[1]),
                      'mi_real_full': float(mi_real_full), 'motion_full': float(motion_full),
                      'mi_real': float(obj_values[5]), 'mi': float(obj_values[4]),
                      'ce': float(obj_values[2]),
                      'minus_ge': float(obj_values[3]),
                      'motion': float(obj_values[6]), 'norm_q': float(obj_values[7]),
                      'norm_q_mixed': float(obj_values[10]),
                      'norm_q_dot': float(obj_values[8]),
                      'norm_q_dot_dot': float(obj_values[9]),
                      'norm_q_dot_dot_dot': float(obj_values[11]),
                      'eps1': self.fe.eps1,
                      'eps2': self.fe.eps2,
                      'eps3': self.fe.eps3}

            # print("FEATURES:")
            # print(features)

            # print("FILTERS:")
            # print(filters)

            # checking errors
            if not np.isfinite(filters).any() or np.isnan(filters).any():
                raise ValueError("Filters contain NaNs or Infinite!")
            if not np.isfinite(features).any() or np.isnan(features).any():
                raise ValueError("Feature maps contain NaNs or Infinite!")

            # save output
            step_save_time = time.time()
            if not self.save_scores_only:
                self.output_stream.save_next(current_img, current_of, features, filters, others)

            step_save_time = time.time() - step_save_time

            # updating rho (print only)
            self.__rho = next_rho

            # stats
            step_time = time.time() - step_time
            self.__elapsed_time = time.time() - self.__start_time
            self.steps = self.steps + 1.0
            self.measured_fps = self.steps / self.__elapsed_time

            if is_night == 1.0:
                self.blink_steps.append(self.steps)

            # saving model (every 1000 steps)
            if int(self.steps) % 1000 == 0:
                self.save()
        else:
            status = False
            step_load_time = time.time() - step_load_time
            step_save_time = 0.0

            # stats
            step_time = time.time() - step_time
            self.__elapsed_time = time.time() - self.__start_time

        return status, step_time, (step_time - step_save_time), (step_time - step_save_time - step_load_time)
