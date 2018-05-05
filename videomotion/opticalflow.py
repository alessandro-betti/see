import cv2
import numpy as np
from gzip import GzipFile


class OpticalFlow:

    def __init__(self, ):
        self.prev_frame = None
        self.frame = None
        self.frame_gray_scale = None
        self.optical_flow = None

    def compute_flow(self, frame, pass_by=False, do_not_update_frame_references=False):

        # --------------------------------------------------------
        # this code disables the optical flow computation
        # (an empty motion field is returned)
        # --------------------------------------------------------
        # if self.optical_flow is None:
        #     a, b, c = frame.shape
        #     self.optical_flow = np.zeros((a, b, 2), frame.dtype)
        # return self.optical_flow
        # --------------------------------------------------------

        if not pass_by and self.frame is not None:
            prev_frame_gray_scale = self.frame_gray_scale
            frame_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not do_not_update_frame_references:
                self.prev_frame = self.frame
                self.frame = frame
                self.frame_gray_scale = frame_gray_scale

            # motion flow is estimated from frame at time t to frame at time t-1 (yes, backward...)
            # then the flow is inverted (changed sign, so now it is going forward),
            # so that we approximate motion from t to t+1 by assuming that
            # the flow has not changed from what was happening in the past
            self.optical_flow = cv2.calcOpticalFlowFarneback(frame_gray_scale,
                                                             prev_frame_gray_scale,
                                                             self.optical_flow,
                                                             pyr_scale=0.4,
                                                             levels=5,  # pyramid levels
                                                             winsize=12,
                                                             iterations=10,
                                                             poly_n=5,
                                                             poly_sigma=1.1,
                                                             flags=0)
            self.optical_flow = -self.optical_flow

            # -------------------------------------------------------------------------------------
            # this is the code I used to qualitatively evaluate the optical flow field
            # when used to rebuild the destination image given the source image and the
            # field itself
            # -------------------------------------------------------------------------------------
            # v = -self.optical_flow
            # a, b = frame_gray_scale.shape
            # guessed_frame = np.zeros((a, b), frame_gray_scale.dtype)
            # for i in range(0, a):
            #     for j in range(0, b):
            #         ii = max(min(int(round(i + v[i][j][1])), a - 1), 0)
            #         jj = max(min(int(round(j + v[i][j][0])), b - 1), 0)
            #         guessed_frame[ii][jj] = frame_gray_scale[i][j]
            # cv2.imwrite("test0REAL.png", prev_frame_gray_scale)
            # cv2.imwrite("test1REAL.png", frame_gray_scale)
            # cv2.imwrite("test0REBUILT.png", guessed_frame)
            # cv2.imwrite("testFLOWA.png", OpticalFlow.draw_flow_map(self.frame, v))
            # cv2.imwrite("testFLOWB.png", OpticalFlow.draw_flow_lines(self.frame, v, line_step=5))
            # -------------------------------------------------------------------------------------

        else:
            if self.frame is None:
                a, b, c = frame.shape
                self.optical_flow = np.zeros((a, b, 2), frame.dtype)

            self.frame = frame
            self.frame_gray_scale = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        return self.optical_flow

    @staticmethod
    def load_flow(file_name):

        # loading optical flow
        file = GzipFile(file_name, 'rb')
        optical_flow = np.load(file)
        file.close()

        return optical_flow

    @staticmethod
    def save_flow(file_name, optical_flow, prev_frame, save_visualizations):

        # saving optical flow
        with GzipFile(file_name, 'wb') as file:
            np.save(file, optical_flow)

        # saving a couple of images to show the computed optical flow
        if save_visualizations:
            cv2.imwrite(file_name + ".lines.png", OpticalFlow.draw_flow_lines(prev_frame, optical_flow))
            cv2.imwrite(file_name + ".map.png", OpticalFlow.draw_flow_map(prev_frame, optical_flow))

    @staticmethod
    def draw_flow_lines(frame, optical_flow, line_step=16, line_color=(0, 255, 0)):
        frame_with_lines = frame.copy()
        line_color = (line_color[2], line_color[1], line_color[0])

        for y in range(0, optical_flow.shape[0], line_step):
            for x in range(0, optical_flow.shape[1], line_step):
                fx, fy = optical_flow[y, x]
                cv2.line(frame_with_lines, (x, y), (int(x + fx), int(y + fy)), line_color)
                cv2.circle(frame_with_lines, (x, y), 1, line_color, -1)

        return frame_with_lines

    @staticmethod
    def draw_flow_map(prev_frame, optical_flow):
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frame_flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame_flow_map
