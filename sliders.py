import cv2


class Sliders:
    def __init__(self):
        cv2.namedWindow("Sliders", flags=cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Color", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Gray", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Region", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Canny", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Hough", "Sliders", 1, 1, lambda x: x)
        cv2.createTrackbar("Canny LTH", "Sliders", 100, 500, lambda x: x)
        cv2.createTrackbar("Canny HTH", "Sliders", 200, 500, lambda x: x)
        cv2.createTrackbar("Hough TH", "Sliders", 100, 500, lambda x: x)
        cv2.createTrackbar("Hough MINL", "Sliders", 50, 500, lambda x: x)
        cv2.createTrackbar("Hough MAXG", "Sliders", 50, 500, lambda x: x)

        self.apply_color = cv2.getTrackbarPos("Color", "Sliders")
        self.apply_gray = cv2.getTrackbarPos("Gray", "Sliders")
        self.apply_region = cv2.getTrackbarPos("Region", "Sliders")
        self.apply_canny = cv2.getTrackbarPos("Canny", "Sliders")
        self.apply_hough = cv2.getTrackbarPos("Hough", "Sliders")
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
            self.canny_low_th,
            self.canny_high_th,
            self.hough_th,
            self.hough_min_line_length,
            self.hough_max_line_gap,
        )
