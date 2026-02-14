import cv2 as cv
import numpy as np


class LaneFollower:

    def __init__(self):
        # Sliding window params
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def process(self, frame):
        """
        Receives BGR frame.
        Returns lateral pixel error (car center - lane center).
        """

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv.threshold(blur, 120, 255, cv.THRESH_BINARY)

        h, w = binary.shape

        # ROI
        mask = np.zeros_like(binary)
        roi_vertices = np.array([[
            (int(0.2 * w), h),
            (int(0.35 * w), int(0.6 * h)),
            (int(0.6 * w), int(0.6 * h)),
            (int(0.85 * w), h)
        ]], dtype=np.int32)

        cv.fillPoly(mask, roi_vertices, 255)
        roi = cv.bitwise_and(binary, mask)

        # Perspective transform
        src = np.float32([
            [0.35 * w, 0.6 * h],
            [0.6 * w, 0.6 * h],
            [0.85 * w, h],
            [0.2 * w, h]
        ])
        dst = np.float32([
            [0.25 * w, 0],
            [0.75 * w, 0],
            [0.75 * w, h],
            [0.25 * w, h]
        ])

        M = cv.getPerspectiveTransform(src, dst)
        warped = cv.warpPerspective(roi, M, (w, h))

        # Histogram
        histogram = np.sum(warped[h//2:, :], axis=0)
        midpoint = histogram.shape[0] // 2

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = h // self.nwindows
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):

            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin

            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(rightx) == 0:
            return 0  # no detection

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, h-1, h)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        car_position = w / 2
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2

        error_pixels = car_position - lane_center

        return error_pixels
