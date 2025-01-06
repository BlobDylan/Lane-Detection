import cv2
import numpy as np
import argparse
import consts
import sliders


def get_perspective_transform_matrix(frame):
    rows, cols = frame.shape[:2]
    src = np.float32(
        [
            [cols * 0.3, rows * 0.75],
            [cols * 0.7, rows * 0.75],
            [cols, rows],
            [0, rows],
        ]
    )
    dst = np.float32(
        [
            [0, 0],
            [cols, 0],
            [cols, rows],
            [0, rows],
        ]
    )
    return cv2.getPerspectiveTransform(src, dst)


def draw_perspective_lines(frame):
    rows, cols = frame.shape[:2]
    src = np.float32(
        [
            [cols * 0.3, rows * 0.75],
            [cols * 0.7, rows * 0.75],
            [cols, rows],
            [0, rows],
        ]
    )
    for i in range(4):
        cv2.line(
            frame,
            (int(src[i][0]), int(src[i][1])),
            (int(src[(i + 1) % 4][0]), int(src[(i + 1) % 4][1])),
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
    return frame


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
    top_left = [int(cols * 0.2), 0]

    bottom_right = [cols, rows]
    top_right = [int(cols * 0.8), 0]

    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]],
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
    base_layer=1,
    region_of_interest_mask=None,
    perspective_transform_matrix=None,
    inverse_perspective_transform_matrix=None,
    apply_color=1,
    apply_threshold=1,
    apply_gray=1,
    apply_region=1,
    apply_perspective=1,
    apply_canny=1,
    apply_hough=1,
    gaussian_blur_kernel_size=consts.DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE,
    canny_low_th=consts.DEFAULT_CANNY_LOW_TH,
    canny_high_th=consts.DEFAULT_CANNY_HIGH_TH,
    dilate_kernel_size=consts.DEFAULT_DILATE_KERNEL_SIZE,
    hough_th=consts.DEFAULT_HOUGH_TH,
    hough_min_line_length=consts.DEFAULT_HOUGH_MIN_LINE_LENGTH,
    hough_max_line_gap=consts.DEFAULT_HOUGH_MAX_LINE_GAP,
):
    original_frame = frame.copy()
    # applying GaussianBlur
    frame = cv2.GaussianBlur(
        frame, (2 * gaussian_blur_kernel_size + 1, 2 * gaussian_blur_kernel_size + 1), 0
    )

    # applying color mask
    if apply_color:
        color_mask = get_color_mask(frame)
        frame = cv2.bitwise_and(frame, frame, mask=color_mask)

    # converting to grayscale
    if apply_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # applying threshold
    if apply_threshold:
        _, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

    # applying perspective transform
    if apply_perspective and perspective_transform_matrix is not None:
        frame = cv2.warpPerspective(
            frame, perspective_transform_matrix, (frame.shape[1], frame.shape[0])
        )
    else:
        frame = draw_perspective_lines(frame)

    # applying region of interest mask
    if apply_region and region_of_interest_mask is not None:
        frame = cv2.bitwise_and(frame, region_of_interest_mask)

    # applying Canny
    if apply_canny:
        frame = cv2.Canny(frame, canny_low_th, canny_high_th)

    # Dilate the image to get a better result
    kernel = np.ones((dilate_kernel_size * 2 + 1, dilate_kernel_size * 2 + 1), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=1)

    # applying HoughLines
    if apply_hough and not apply_canny and not apply_gray:
        return "HoughLines can only be applied after Canny or Gray", None

    if apply_hough:
        hough_linesp = cv2.HoughLinesP(
            frame,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_th,
            minLineLength=hough_min_line_length,
            maxLineGap=hough_max_line_gap,
        )
        output_frame = original_frame if base_layer else frame
        if hough_linesp is not None:
            for line in hough_linesp:
                x1, y1, x2, y2 = line[0]
                # Apply inverse perspective transform to the line endpoints
                points = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(
                    -1, 1, 2
                )
                if (
                    apply_perspective
                    and inverse_perspective_transform_matrix is not None
                ):
                    transformed_points = cv2.perspectiveTransform(
                        points, inverse_perspective_transform_matrix
                    )
                else:
                    transformed_points = points
                x1, y1 = transformed_points[0][0]
                x2, y2 = transformed_points[1][0]
                cv2.line(
                    original_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

    # resizing the frame just so it fits in the screen for display purposes.
    output_frame = cv2.resize(
        output_frame, (consts.DISPLAY_WIDTH, consts.DISPLAY_HEIGHT)
    )

    return None, output_frame


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
        perspective_transform_matrix = None
        inverse_perspective_transform_matrix = None
        while True:
            ret, input_frame = cap.read()
            if not ret:
                break

            # get the current trackbar values if in slider mode
            if args.sliders:
                (
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
                ) = sliders_instance.get_values()
            else:
                base_layer = 1
                apply_color = 1
                apply_gray = 1
                apply_threshold = 1
                apply_region = 1
                apply_perspective = 1
                apply_canny = 1
                apply_hough = 1
                gaussian_blur_kernel_size = consts.DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE
                canny_low_th = consts.DEFAULT_CANNY_LOW_TH
                canny_high_th = consts.DEFAULT_CANNY_HIGH_TH
                dilate_kernel_size = consts.DEFAULT_DILATE_KERNEL_SIZE
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

            # get the perspective transform matrix only once.
            if perspective_transform_matrix is None:
                perspective_transform_matrix = get_perspective_transform_matrix(
                    input_frame
                )
                inverse_perspective_transform_matrix = np.linalg.inv(
                    perspective_transform_matrix
                )

            # detect lanes in the frame
            err, output_frame = detect_lanes_in_frame(
                frame=input_frame,
                base_layer=base_layer,
                region_of_interest_mask=region_of_interest_mask,
                perspective_transform_matrix=perspective_transform_matrix,
                inverse_perspective_transform_matrix=inverse_perspective_transform_matrix,
                apply_color=apply_color,
                apply_gray=apply_gray,
                apply_threshold=apply_threshold,
                apply_region=apply_region,
                apply_perspective=apply_perspective,
                apply_canny=apply_canny,
                apply_hough=apply_hough,
                gaussian_blur_kernel_size=gaussian_blur_kernel_size,
                canny_low_th=canny_low_th,
                canny_high_th=canny_high_th,
                dilate_kernel_size=dilate_kernel_size,
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
