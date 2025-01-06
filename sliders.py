import cv2
import consts


class Sliders:
    def __init__(self):
        cv2.namedWindow("Sliders", flags=cv2.WINDOW_NORMAL)
        trackbars = {
            "Base Layer": (1, 1),
            "Color": (1, 1),
            "Gray": (1, 1),
            "Threshold": (1, 1),
            "Region": (1, 1),
            "Perspective": (1, 1),
            "Canny": (1, 1),
            "Hough": (1, 1),
            "Gaussian K": (consts.DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE, 50),
            "Canny LTH": (consts.DEFAULT_CANNY_LOW_TH, 500),
            "Canny HTH": (consts.DEFAULT_CANNY_HIGH_TH, 500),
            "Dilate K": (consts.DEFAULT_DILATE_KERNEL_SIZE, 5),
            "Hough TH": (consts.DEFAULT_HOUGH_TH, 500),
            "Hough MINL": (consts.DEFAULT_HOUGH_MIN_LINE_LENGTH, 500),
            "Hough MAXG": (consts.DEFAULT_HOUGH_MAX_LINE_GAP, 500),
        }

        # Create trackbars
        for name, (default, max_value) in trackbars.items():
            cv2.createTrackbar(name, "Sliders", default, max_value, lambda x: x)

    def get_values(self):
        base_layer = cv2.getTrackbarPos("Base Layer", "Sliders")
        apply_color = cv2.getTrackbarPos("Color", "Sliders")
        apply_gray = cv2.getTrackbarPos("Gray", "Sliders")
        apply_threshold = cv2.getTrackbarPos("Threshold", "Sliders")
        apply_region = cv2.getTrackbarPos("Region", "Sliders")
        apply_perspective = cv2.getTrackbarPos("Perspective", "Sliders")
        apply_canny = cv2.getTrackbarPos("Canny", "Sliders")
        apply_hough = cv2.getTrackbarPos("Hough", "Sliders")
        gaussian_blur_kernel_size = cv2.getTrackbarPos("Gaussian K", "Sliders")
        canny_low_th = cv2.getTrackbarPos("Canny LTH", "Sliders")
        canny_high_th = cv2.getTrackbarPos("Canny HTH", "Sliders")
        dilate_kernel_size = cv2.getTrackbarPos("Dilate K", "Sliders")
        hough_th = cv2.getTrackbarPos("Hough TH", "Sliders")
        hough_min_line_length = cv2.getTrackbarPos("Hough MINL", "Sliders")
        hough_max_line_gap = cv2.getTrackbarPos("Hough MAXG", "Sliders")
        return (
            base_layer,
            apply_color,
            apply_gray,
            apply_threshold,
            apply_region,
            apply_perspective,
            apply_canny,
            apply_hough,
            gaussian_blur_kernel_size,
            canny_low_th,
            canny_high_th,
            dilate_kernel_size,
            hough_th,
            hough_min_line_length,
            hough_max_line_gap,
        )
