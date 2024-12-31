import cv2
import consts


class Sliders:
    def __init__(self):
        cv2.namedWindow("Sliders", flags=cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Color", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Gray", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Region", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Canny", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Hough", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar(
            "Bilateral d", "Sliders", consts.DEFAULT_BILATERALFILTER_D, 9, lambda x: x
        )
        cv2.createTrackbar(
            "Bilateral SC",
            "Sliders",
            consts.DEFAULT_BILATERALFILTER_SIGMA_COLOR,
            300,
            lambda x: x,
        )
        cv2.createTrackbar(
            "Bilateral SS",
            "Sliders",
            consts.DEFAULT_BILATERALFILTER_SIGMA_SPACE,
            300,
            lambda x: x,
        )
        cv2.createTrackbar(
            "Gaussian K",
            "Sliders",
            consts.DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE,
            20,
            lambda x: x,
        )
        cv2.createTrackbar(
            "Canny LTH", "Sliders", consts.DEFAULT_CANNY_LOW_TH, 500, lambda x: x
        )
        cv2.createTrackbar(
            "Canny HTH", "Sliders", consts.DEFAULT_CANNY_HIGH_TH, 500, lambda x: x
        )
        cv2.createTrackbar(
            "Hough TH", "Sliders", consts.DEFAULT_HOUGH_TH, 500, lambda x: x
        )
        cv2.createTrackbar(
            "Hough MINL",
            "Sliders",
            consts.DEFAULT_HOUGH_MIN_LINE_LENGTH,
            500,
            lambda x: x,
        )
        cv2.createTrackbar(
            "Hough MAXG", "Sliders", consts.DEFAULT_HOUGH_MAX_LINE_GAP, 500, lambda x: x
        )

        self.apply_color = cv2.getTrackbarPos("Color", "Sliders")
        self.apply_gray = cv2.getTrackbarPos("Gray", "Sliders")
        self.apply_region = cv2.getTrackbarPos("Region", "Sliders")
        self.apply_canny = cv2.getTrackbarPos("Canny", "Sliders")
        self.apply_hough = cv2.getTrackbarPos("Hough", "Sliders")
        self.bilateral_filter_d = cv2.getTrackbarPos("Bilateral d", "Sliders")
        self.bilateral_filter_sigma_color = cv2.getTrackbarPos(
            "Bilateral SC", "Sliders"
        )
        self.bilateral_filter_sigma_space = cv2.getTrackbarPos(
            "Bilateral SS", "Sliders"
        )
        self.gaussian_blur_kernel_size = cv2.getTrackbarPos("Gaussian K", "Sliders")
        self.canny_low_th = cv2.getTrackbarPos("Canny LTH", "Sliders")
        self.canny_high_th = cv2.getTrackbarPos("Canny HTH", "Sliders")
        self.hough_th = cv2.getTrackbarPos("Hough TH", "Sliders")
        self.hough_min_line_length = cv2.getTrackbarPos("Hough MINL", "Sliders")
        self.hough_max_line_gap = cv2.getTrackbarPos("Hough MAXG", "Sliders")

    def get_values(self):
        self.apply_color = cv2.getTrackbarPos("Color", "Sliders")
        self.apply_gray = cv2.getTrackbarPos("Gray", "Sliders")
        self.apply_region = cv2.getTrackbarPos("Region", "Sliders")
        self.apply_canny = cv2.getTrackbarPos("Canny", "Sliders")
        self.apply_hough = cv2.getTrackbarPos("Hough", "Sliders")
        self.bilateral_filter_d = cv2.getTrackbarPos("Bilateral d", "Sliders")
        self.bilateral_filter_sigma_color = cv2.getTrackbarPos(
            "Bilateral SC", "Sliders"
        )
        self.bilateral_filter_sigma_space = cv2.getTrackbarPos(
            "Bilateral SS", "Sliders"
        )
        self.gaussian_blur_kernel_size = cv2.getTrackbarPos("Gaussian K", "Sliders")
        self.canny_low_th = cv2.getTrackbarPos("Canny LTH", "Sliders")
        self.canny_high_th = cv2.getTrackbarPos("Canny HTH", "Sliders")
        self.hough_th = cv2.getTrackbarPos("Hough TH", "Sliders")
        self.hough_min_line_length = cv2.getTrackbarPos("Hough MINL", "Sliders")
        self.hough_max_line_gap = cv2.getTrackbarPos("Hough MAXG", "Sliders")
        return (
            self.apply_color,
            self.apply_gray,
            self.apply_region,
            self.apply_canny,
            self.apply_hough,
            self.bilateral_filter_d,
            self.bilateral_filter_sigma_color,
            self.bilateral_filter_sigma_space,
            self.gaussian_blur_kernel_size,
            self.canny_low_th,
            self.canny_high_th,
            self.hough_th,
            self.hough_min_line_length,
            self.hough_max_line_gap,
        )
