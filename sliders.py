import cv2


class Sliders:
    def __init__(self):
        cv2.namedWindow("Trackbars", flags=cv2.WINDOW_GUI_NORMAL)
        cv2.createTrackbar("Region", "Trackbars", 1, 1, lambda x: x)
        cv2.createTrackbar("Canny", "Trackbars", 1, 1, lambda x: x)
        cv2.createTrackbar("Canny LTH", "Trackbars", 100, 500, lambda x: x)
        cv2.createTrackbar("Canny HTH", "Trackbars", 200, 500, lambda x: x)

        self.apply_region = cv2.getTrackbarPos("Region", "Trackbars")
        self.apply_canny = cv2.getTrackbarPos("Canny", "Trackbars")
        self.canny_low_th = cv2.getTrackbarPos("Canny LTH", "Trackbars")
        self.canny_high_th = cv2.getTrackbarPos("Canny HTH", "Trackbars")

    def get_values(self):
        self.apply_region = cv2.getTrackbarPos("Region", "Trackbars")
        self.apply_canny = cv2.getTrackbarPos("Canny", "Trackbars")
        self.canny_low_th = cv2.getTrackbarPos("Canny LTH", "Trackbars")
        self.canny_high_th = cv2.getTrackbarPos("Canny HTH", "Trackbars")
        return (
            self.apply_region,
            self.apply_canny,
            self.canny_low_th,
            self.canny_high_th,
        )
