import sys
import os
import cv2
import shutil
from glob import glob
from opticalflow import OpticalFlow
import numpy as np
from gzip import GzipFile
import json
from decimal import Decimal, ROUND_HALF_UP


class InputStream:
    files_per_folder = 100   # global parameter

    def __init__(self, input_element, input_is_video, blur):
        self.input = input_element
        self.input_is_video = input_is_video
        self.optical_flow_processor = OpticalFlow()
        self.__files_per_folder = self.files_per_folder
        self.w = -1
        self.h = -1
        self.c = -1
        self.frames = -1
        self.max_frames = -1
        self.fps = -1
        self.__blur = blur
        self.__req_w = -1
        self.__req_h = -1
        self.__req_fps = -1
        self.__force_gray = False
        self.__video_capture = None
        self.__last_returned_frame = None
        self.__last_returned_frame_number = 0
        self.__last_returned_of = None
        self.__last_returned_time = 0.0
        self.__getinfo()

    def is_video(self):
        return self.input_is_video

    def is_folder(self):
        return not self.input_is_video

    def reset(self):
        w = self.__req_w
        h = self.__req_h
        fps = self.__req_fps
        force_gray = self.__force_gray
        max_frames = self.max_frames
        self.__init__(self.input, self.input_is_video, self.__blur)
        self.set_options(w, h, fps, force_gray, max_frames)

    def set_options(self, w, h, fps, force_gray, max_frames):
        self.__req_w = w
        self.__req_h = h
        self.__req_fps = fps
        self.__force_gray = force_gray
        self.max_frames = max_frames

    def set_last_frame_and_time(self, frame, seconds):
        self.__last_returned_frame_number = frame - 1
        if self.is_video():
            self.__video_capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)

    def get_last_frame_number(self):
        return self.__last_returned_frame_number

    def get_last_frame_time(self):
        return self.__last_returned_time / 1000.0

    def get_next(self, blur_factor=0.0, t=-1, sample_only=False):
        img = None
        next_time = None
        compute_motion = True
        f = None

        # check
        if 0 < self.max_frames <= self.__last_returned_frame_number:
            return None, None

        # opening stream (if not already opened)
        if self.is_video():
            if self.__video_capture is None or not self.__video_capture.isOpened():
                if self.input != "0":
                    self.__video_capture = cv2.VideoCapture(self.input)
                    self.__video_capture.set(cv2.CAP_PROP_POS_MSEC, self.__last_returned_time)
                else:
                    self.__video_capture = cv2.VideoCapture(0)
        else:
            f = self.__last_returned_frame_number

        # setting time for the next frame
        if self.input != "0":
            if t > 0.0:
                next_time = t * 1000.0
                if next_time >= self.__last_returned_time:
                    if self.__video_capture is not None:
                        self.__video_capture.set(cv2.CAP_PROP_POS_MSEC, next_time)
                    f = Decimal(t * float(self.__req_fps)).quantize(0, ROUND_HALF_UP)
                else:
                    raise IOError("Cannot seek back in time!")
            else:
                if self.__req_fps != self.fps and self.__req_fps > 0.0:
                    next_time = self.__last_returned_time + (1000.0 / float(self.__req_fps))

                    # detect changes when asking for a number of FPS than is bigger than the original one
                    if self.__req_fps > self.fps and self.__last_returned_frame_number > 0:
                        next_frame_original_fps = (next_time * float(self.fps)) / 1000.0
                        cur_frame_original_fps = (self.__last_returned_time * float(self.fps)) / 1000.0
                        if Decimal(next_frame_original_fps).quantize(0, ROUND_HALF_UP) == \
                                Decimal(cur_frame_original_fps).quantize(0, ROUND_HALF_UP):
                            compute_motion = False

                    if self.__video_capture is not None:
                        self.__video_capture.set(cv2.CAP_PROP_POS_MSEC, next_time)
                    f = Decimal(next_time * float(self.__req_fps)).quantize(0, ROUND_HALF_UP)

        # getting a new frame
        if self.is_video():
            ret_val, img = self.__video_capture.read()

            if not ret_val:
                if self.input != "0" and self.__req_fps > 0 and \
                        self.__req_fps != self.fps and next_time == self.__video_capture.get(cv2.CAP_PROP_POS_MSEC):
                    raise IOError("Unable to capture frames from video!")
                elif self.input != "0" and self.__req_fps <= 0 and \
                        self.__video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO) < 1.0:
                    raise IOError("Unable to capture frames from video!")
                else:
                    self.__video_capture.release()
                    img = None  # reached the end of video

            if img is None:
                return None, None

        elif self.is_folder():
            n_folder = int(f / self.__files_per_folder) + 1
            n_file = (f + 1) - ((n_folder - 1) * self.__files_per_folder)

            folder_name = format(n_folder, '08d')
            file_name = format(n_file, '03d')

            if os.path.exists(self.input + os.sep + "frames" + os.sep + folder_name + os.sep + file_name + ".png"):
                img = cv2.imread(self.input + os.sep + "frames" + os.sep + folder_name + os.sep + file_name + ".png")
            else:
                return None, None

        # rescaling
        if (self.__req_w > 0 and self.__req_h > 0) and (self.__req_w != self.w or self.__req_h != self.h):
            img = cv2.resize(img, (self.__req_w, self.__req_h))

        # blurring
        if blur_factor > 0.0:
            if blur_factor > 1.0:
                raise ValueError("Invalid blur factor (it must be in [0,1]): " + str(blur_factor))
            ww, hh, dd = img.shape
            kernel_size = int(round(float(min(ww, hh)) * 0.25 * blur_factor))
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            if self.__blur:
                img = ((1.0 - blur_factor) * cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)).astype(np.uint8)
            else:
                img = ((1.0 - blur_factor) * img).astype(np.uint8)
        else:
            if blur_factor < 0.0:
                raise ValueError("Invalid blur factor (it must be in [0,1]): " + str(blur_factor))

        # computing optical flow
        if compute_motion or self.__last_returned_of is None:
            of = self.optical_flow_processor.compute_flow(img,
                                                          pass_by=(self.__last_returned_of is None),
                                                          do_not_update_frame_references=sample_only)
        else:
            of = self.__last_returned_of

        if not sample_only:
            if next_time is None:
                next_time = self.__last_returned_time + (1000.0 / float(self.fps))

            self.__last_returned_frame_number = self.__last_returned_frame_number + 1
            self.__last_returned_time = next_time
            self.__last_returned_frame = img
            self.__last_returned_of = of

            # open CV is buggy in counting frames...
            if self.__last_returned_frame_number > self.frames:
                self.frames = self.__last_returned_frame_number

        if not self.__force_gray:
            return img, of
        else:
            return np.reshape(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (self.__req_h, self.__req_w, 1)), of

    def __getinfo(self):
        if self.is_video():
            if self.input != "0":
                video = cv2.VideoCapture(self.input)
            else:
                video = cv2.VideoCapture(0)

            if video.isOpened():
                fps = video.get(cv2.CAP_PROP_FPS)  # float
                frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                if self.input != "0":
                    frames = int(frames)
                else:
                    frames = sys.maxsize  # dummy

                ret_val, img = video.read()

                if ret_val:
                    h, w, c = img.shape
                    w = int(w)
                    h = int(h)
                    c = int(c)
                else:
                    raise IOError("Error while trying to grab a frame from: ", self.input)
            else:
                raise IOError("Cannot open: ", self.input)

            self.w = w
            self.h = h
            self.c = c
            self.frames = frames
            self.fps = fps

            if self.frames <= 0:
                raise ValueError("Invalid frame count: " + str(self.frames))
            if self.fps <= 0:
                raise ValueError("Invalid FPS count: " + str(self.fps))
            if self.w <= 0 or self.h <= 0:
                raise ValueError("Invalid resolution: " + str(self.w) + "x" + str(self.h))

        elif self.is_folder():
            first_file = ''
            dirs = glob(self.input + os.sep + "frames" + os.sep + "*" + os.sep)

            if dirs is not None and len(dirs) > 0:
                dirs.sort()

                n = len(dirs) - 2  # discarding '.' and '..'
                i = 1

                for d in dirs:
                    if not os.path.isdir(d):
                        continue
                    d = os.path.basename(os.path.dirname(d))
                    if d == '.' or d == '..':
                        continue

                    folder_name = format(i, '08d')
                    if folder_name != d:
                        raise ValueError("Invalid/unexpected folder: " + self.input + os.sep + "frames" + os.sep + d)

                    files = glob(self.input + os.sep + "frames" + os.sep + d + os.sep + "*.png")
                    files.sort()
                    j = 1

                    if i < n and len(files) != self.__files_per_folder:
                        raise ValueError("Invalid/unexpected number of files in: "
                                         + self.input + os.sep + "frames" + os.sep + d)

                    for f in files:
                        file_name = format(j, '03d')
                        f = os.path.basename(f)
                        if file_name + ".png" != f:
                            raise ValueError("Invalid/unexpected file '" + f + "' in: "
                                             + self.input + os.sep + "frames" + os.sep + d)
                        j = j + 1

                    if len(first_file) == 0:
                        files.sort()
                        first_file = files[0]
                        self.frames = 0

                    self.frames = self.frames + len(files)

                    i = i + 1

                img = cv2.imread(first_file)
                h, w, c = img.shape

                self.w = int(w)
                self.h = int(h)
                self.c = int(c)
                self.fps = -1

                if self.frames <= 0:
                    raise ValueError("Invalid frame count: " + str(self.frames))
                if self.w <= 0 or self.h <= 0:
                    raise ValueError("Invalid resolution: " + str(self.w) + "x" + str(self.h))

                try:
                    opts = json.load(open(self.input + os.sep + "options.txt"))
                    self.fps = float(opts['fps'])
                except (ValueError, IOError):
                    raise IOError("Options file is missing/unreadable! (needed to read the FPS value): "
                                  + self.input + os.sep + "options.txt")
            else:
                raise ValueError("No frames in: " + self.input + os.sep + "frames" + os.sep)


class OutputStream:
    files_per_folder = InputStream.files_per_folder

    def __init__(self, folder, purge_existing_data):
        self.folder = folder
        self.optical_flow_processor = OpticalFlow()
        self.__files_per_folder = self.files_per_folder
        self.__last_saved_frame_number = 0
        self.__options = {}
        self.create_folders(purge_existing_data)

    def create_folders(self, purge_existing_data):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        elif purge_existing_data:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        if not os.path.exists(self.folder + os.sep + 'frames'):
            os.makedirs(self.folder + os.sep + 'frames')
        if not os.path.exists(self.folder + os.sep + 'motion'):
            os.makedirs(self.folder + os.sep + 'motion')
        if not os.path.exists(self.folder + os.sep + 'features'):
            os.makedirs(self.folder + os.sep + 'features')
        if not os.path.exists(self.folder + os.sep + 'filters'):
            os.makedirs(self.folder + os.sep + 'filters')
        if not os.path.exists(self.folder + os.sep + 'others'):
            os.makedirs(self.folder + os.sep + 'others')

    def save_next(self, img, of, features, filters, others):

        # getting the right folder and file ID
        f = self.__last_saved_frame_number
        n_folder = int(f / self.__files_per_folder) + 1
        n_file = (f + 1) - ((n_folder - 1) * self.__files_per_folder)

        folder_name = format(n_folder, '08d')
        file_name = format(n_file, '03d')

        # creating the internal folders, if needed
        if not os.path.isdir(self.folder + os.sep + "frames" + os.sep + folder_name):
            os.makedirs(self.folder + os.sep + "frames" + os.sep + folder_name)
        if not os.path.isdir(self.folder + os.sep + "motion" + os.sep + folder_name):
            os.makedirs(self.folder + os.sep + "motion" + os.sep + folder_name)
        if not os.path.isdir(self.folder + os.sep + "features" + os.sep + folder_name):
            os.makedirs(self.folder + os.sep + "features" + os.sep + folder_name)
        if not os.path.isdir(self.folder + os.sep + "filters" + os.sep + folder_name):
            os.makedirs(self.folder + os.sep + "filters" + os.sep + folder_name)
        if not os.path.isdir(self.folder + os.sep + "others" + os.sep + folder_name):
            os.makedirs(self.folder + os.sep + "others" + os.sep + folder_name)

        # saving frame
        if not os.path.exists(self.folder + os.sep + "frames" + os.sep + folder_name + os.sep + file_name + ".png"):
            cv2.imwrite(self.folder + os.sep + "frames" + os.sep + folder_name + os.sep + file_name + ".png", img)

        # saving motion field
        if not os.path.exists(self.folder + os.sep + "motion" + os.sep + folder_name + os.sep + file_name + ".of"):
            self.optical_flow_processor.save_flow(self.folder + os.sep + "motion" + os.sep + folder_name + os.sep
                                                  + file_name + ".of", of, img, False)

        # saving features
        with GzipFile(self.folder + os.sep + "features" + os.sep + folder_name + os.sep + file_name +
                      ".feat", 'wb') as file:
                np.save(file, features)

        # saving filters
        with GzipFile(self.folder + os.sep + "filters" + os.sep + folder_name + os.sep + file_name +
                      ".fil", 'wb') as file:
                np.save(file, filters)

        # saving others
        f = open(self.folder + os.sep + "others" + os.sep + folder_name + os.sep + file_name + ".txt", 'w')
        if f is None or not f or f.closed:
            raise IOError("Cannot access: " + self.folder + os.sep + "others" + os.sep + folder_name
                          + os.sep + file_name + ".txt")
        json.dump(others, f, indent=4)
        f.close()

        self.__last_saved_frame_number = self.__last_saved_frame_number + 1

    def save_option(self, opt, value):
        self.__options[opt] = value

        f = open(self.folder + os.sep + "options.txt", "w")
        if f is None or not f or f.closed:
            raise IOError("Cannot access: " + self.folder + os.sep + "options.txt")
        json.dump(self.__options, f, indent=4)
        f.close()

    def set_last_frame(self, last_frame_number):
        self.__last_saved_frame_number = last_frame_number - 1
