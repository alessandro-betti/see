import numpy as np
import time
from feature_extractor import FeatureExtractor
from utils import err, out
import json


class Worker:

    def __init__(self, input_stream, output_stream, w=-1, h=-1, fps=-1, frames=-1,
                 force_gray=False, repetitions=1, options={}, resume=False):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.w = w
        self.h = h
        self.fps = fps
        self.frames = frames
        self.force_gray = force_gray
        self.repetitions = repetitions
        self.__completed_repetitions = 0
        self.__previous_img = None
        self.__start_time = None
        self.__elapsed_time = None
        self.__rho = 0.0
        self.steps = 0.0
        self.measured_fps = 0.0
        self.save_scores_only = options['save_scores_only']
        self.fe = FeatureExtractor(w, h, options, resume)  # here is the TensorFlow based feature extractor!

        if resume:
            out("RESUMING...")
            self.load()

    def close(self):
        self.fe.close()

    def save(self):
        self.fe.save()

        info = {'steps': self.steps, 'frame': self.input_stream.get_frame_number()}

        f = open(self.fe.save_path + ".info.txt", "w")
        if f is None or not f or f.closed:
            raise IOError("Cannot access: " + self.fe.save_path + ".info.txt")
        json.dump(info, f, indent=4)
        f.close()

    def load(self):
        f = open(self.fe.save_path + ".info.txt", "r")
        if f is None or not f or f.closed:
            raise IOError("Cannot access: " + self.fe.save_path + ".info.txt")
        info = json.load(f)
        f.close()

        self.steps = info['steps']

        self.fe.load(self.steps)

        self.input_stream.set_frame_number(info['frame'] - 1, self.w, self.h, self.fps, self.frames, self.force_gray)
        current_img, current_of = self.input_stream.get_next(self.w, self.h, self.fps, self.frames, self.force_gray)
        self.__previous_img = current_img

    def run_step(self):

        while True:

            # get time
            status = True
            step_time = time.time()
            if self.__start_time is None:
                self.__start_time = step_time

            # get the frame to process at the next step and the currently needed motion field
            step_load_time = time.time()
            current_img, current_of = self.input_stream.get_next(self.w, self.h, self.fps, self.frames, self.force_gray)

            # fixing the case of the first frame
            if self.__previous_img is None:
                self.__previous_img = current_img.copy()

            # handling repetitions
            if current_img is None:
                self.__completed_repetitions = self.__completed_repetitions + 1
                if self.__completed_repetitions < self.repetitions:
                    self.input_stream.reset()
                    self.__previous_img = None
                else:
                    break
            else:
                break

        # check if the stream has ended
        if current_img is not None:
            step_load_time = time.time() - step_load_time

            # extracting features
            features, filters, mi, mi_real, ce, minus_ge, sum_to_one, negativeness, motion, norm_q, norm_q_dot, norm_q_dot_dot, \
                norm_q_dot_dot_dot, norm_q_mixed, all_terms, is_night, \
                next_rho = self.fe.run_step(self.__previous_img, current_img, current_of)

            # output-info
            if is_night == 1.0:
                light = "night"
            else:
                light = "day"

            out("\t[status=" + light + ", rho=" + str(self.__rho) + ", action_approx=" + str(all_terms)
                + ",\n\t mi_real=" + str(mi_real) + ", mi=" + str(mi) + ", ce=" + str(ce) + ", minus_ge=" + str(minus_ge)
                + ", sum1=" + str(sum_to_one)
                + ", negativeness=" + str(negativeness) + ", motion=" + str(motion) + ",\n\t norm_q="
                + str(norm_q) + ", norm_q'=" + str(norm_q_dot) + ", norm_q''=" + str(norm_q_dot_dot)
                + ", norm_q'''=" + str(norm_q_dot_dot_dot)
                + ", q'q''=" + str(norm_q_mixed) + "]")

            others = {'mi': float(mi), 'mi_real': float(mi_real), 'ce': float(ce), 'minus_ge': float(minus_ge), 'sum_to_one': float(sum_to_one),
                      'negativeness': float(negativeness),
                      'motion': float(motion), 'norm_q': float(norm_q), 'norm_q_dot': float(norm_q_dot),
                      'norm_q_dot_dot': float(norm_q_dot_dot),
                      'norm_q_dot_dot_dot': float(norm_q_dot_dot_dot), 'norm_q_mixed': float(norm_q_mixed),
                      'all': float(all_terms),
                      'is_night': float(is_night), 'rho': float(self.__rho)}

            # print("FEATURES:")
            # print(features)

            # print("FILTERS:")
            # print(filters)

            # checking errors
            if not np.isfinite(filters).any() or np.isnan(filters).any():
                raise ValueError("Filters contain NaNs or Infs!")
            if not np.isfinite(features).any() or np.isnan(features).any():
                raise ValueError("Feature maps contain NaNs or Infs!")

            # save output
            step_save_time = time.time()
            if not self.save_scores_only:
                self.output_stream.save_next(current_img, current_of, features, filters, others)
            step_save_time = time.time() - step_save_time

            # saving a reference used by the next frame to process
            self.__previous_img = current_img

            # updating rho (print only)
            self.__rho = next_rho

            # stats
            step_time = time.time() - step_time
            self.__elapsed_time = time.time() - self.__start_time
            self.steps = self.steps + 1.0
            self.measured_fps = self.steps / self.__elapsed_time

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

            # releasing the feature extractor
            self.fe.close()

        return status, step_time, (step_time - step_save_time), (step_time - step_save_time - step_load_time)
