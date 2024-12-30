import cv2
import numpy as np
import argparse
import consts
import sliders


def get_region_of_interest_mask(frame, apply_gray=1):
    rows, cols = frame.shape[:2]
    mask_shape = (rows, cols) if apply_gray else (rows, cols, 3)
    mask = np.zeros(mask_shape, dtype=np.uint8)

    if apply_gray:
        ignore_mask_color = 255
    else:
        channel_count = frame.shape[2]
        ignore_mask_color = (255,) * channel_count

    bottom_left = [0, rows]
    middle_left = [0, int(rows * 0.85)]
    top_left = [int(cols * 0.4), int(rows * 0.7)]

    bottom_right = [cols, rows]
    middle_right = [cols, int(rows * 0.85)]
    top_right = [int(cols * 0.6), int(rows * 0.7)]

    vertices = np.array(
        [[bottom_left, middle_left, top_left, top_right, middle_right, bottom_right]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return mask


def get_color_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(
        hsv,
        np.array(consts.DEFAULT_YELLOW_LOWER),
        np.array(consts.DEFAULT_YELLOW_UPPER),
    )
    mask_white = cv2.inRange(
        hsv,
        np.array(consts.DEFAULT_WHITE_LOWER),
        np.array(consts.DEFAULT_WHITE_UPPER),
    )

    mask = cv2.bitwise_or(mask_yellow, mask_white)
    return mask


def detect_lanes_in_frame(
    frame,
    region_of_interest_mask=None,
    apply_color=1,
    apply_gray=1,
    apply_region=1,
    apply_canny=1,
    apply_hough=1,
    canny_low_th=consts.DEFAULT_CANNY_LOW_TH,
    canny_high_th=consts.DEFAULT_CANNY_HIGH_TH,
    hough_th=consts.DEFAULT_HOUGH_TH,
    hough_min_line_length=consts.DEFAULT_HOUGH_MIN_LINE_LENGTH,
    hough_max_line_gap=consts.DEFAULT_HOUGH_MAX_LINE_GAP,
):
    # applying color mask
    if apply_color:
        color_mask = get_color_mask(frame)
        frame = cv2.bitwise_and(frame, frame, mask=color_mask)

    # converting to grayscale
    if apply_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # applying region of interest mask
    if apply_region and region_of_interest_mask is not None:
        frame = cv2.bitwise_and(frame, region_of_interest_mask)

    # applying Canny
    if apply_canny:
        frame = cv2.Canny(frame, canny_low_th, canny_high_th)

    # applying HoughLines
    if apply_hough:
        hough_linesp = cv2.HoughLinesP(
            frame,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_th,
            minLineLength=hough_min_line_length,
            maxLineGap=hough_max_line_gap,
        )
        if hough_linesp is not None:
            for i in range(0, len(hough_linesp)):
                line = hough_linesp[i][0]
                cv2.line(
                    frame,
                    (line[0], line[1]),
                    (line[2], line[3]),
                    (255, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

    # resizing the frame
    frame = cv2.resize(frame, (consts.DISPLAY_WIDTH, consts.DISPLAY_HEIGHT))

    return None, frame


def main(args):
    cap = cv2.VideoCapture(consts.VIDEO_PATH)
    prev_gray = 1
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)
        return

    if args.sliders:
        sliders_instance = sliders.Sliders()

    try:
        region_of_interest_mask = None
        while True:
            ret, input_frame = cap.read()
            if not ret:
                break

            # get the current trackbar values if in slider mode
            if args.sliders:
                (
                    apply_color,
                    apply_gray,
                    apply_region,
                    apply_canny,
                    apply_hough,
                    canny_low_th,
                    canny_high_th,
                    hough_th,
                    hough_min_line_length,
                    hough_max_line_gap,
                ) = sliders_instance.get_values()
            else:
                apply_color = 1
                apply_gray = 1
                apply_region = 1
                apply_canny = 1
                apply_hough = 1
                canny_low_th = consts.DEFAULT_CANNY_LOW_TH
                canny_high_th = consts.DEFAULT_CANNY_HIGH_TH
                hough_th = consts.DEFAULT_HOUGH_TH
                hough_min_line_length = consts.DEFAULT_HOUGH_MIN_LINE_LENGTH
                hough_max_line_gap = consts.DEFAULT_HOUGH_MAX_LINE_GAP

            if apply_gray != prev_gray:
                region_of_interest_mask = None
                prev_gray = apply_gray

            # get the region of interest only once.
            if region_of_interest_mask is None:
                region_of_interest_mask = get_region_of_interest_mask(
                    input_frame, apply_gray
                )

            # detect lanes in the frame
            err, output_frame = detect_lanes_in_frame(
                frame=input_frame,
                region_of_interest_mask=region_of_interest_mask,
                apply_color=apply_color,
                apply_gray=apply_gray,
                apply_region=apply_region,
                apply_canny=apply_canny,
                apply_hough=apply_hough,
                canny_low_th=canny_low_th,
                canny_high_th=canny_high_th,
                hough_th=hough_th,
                hough_min_line_length=hough_min_line_length,
                hough_max_line_gap=hough_max_line_gap,
            )

            # handle errors detecting lanes
            if err:
                print(f"Error: {err}")
                break

            # show the frame
            cv2.imshow("Frame", output_frame)

            # break on 'q' key
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection")
    parser.add_argument(
        "--sliders",
        "-s",
        action="store_true",
        default=False,
        help="Show sliders to adjust parameters",
    )
    args = parser.parse_args()
    main(args)
