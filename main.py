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
    top_left = [int(cols * 0.4), 0]

    bottom_right = [cols, rows]
    top_right = [int(cols * 0.75), 0]

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


def remove_lines_by_angle(lines):
    filtered_lines = []

    left_lane_angle_range = consts.DEFAULT_LEFT_ANGLE_RANGE
    right_lane_angle_range = consts.DEFAULT_RIGHT_ANGLE_RANGE

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 != x1:  # Avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if (
                    left_lane_angle_range[0] <= angle <= left_lane_angle_range[1]
                    or right_lane_angle_range[0] <= angle <= right_lane_angle_range[1]
                ):
                    filtered_lines.append((x1, y1, x2, y2))
    return filtered_lines


def group_close_lines(lines, distance_threshold):
    groups_of_lines = []
    for line in lines:
        x1, _, x2, _ = line
        found_group = False
        for group in groups_of_lines:
            # grouping only by x1 and x2
            # TODO: take average instead of first
            x1g, _, x2g, _ = group[0]
            if (
                abs(x1 - x1g) < distance_threshold
                and abs(x2 - x2g) < distance_threshold
            ):
                group.append(line)
                found_group = True
                break
        if not found_group:
            groups_of_lines.append([line])

    return groups_of_lines


def get_sum_of_lengths(group):
    return sum([np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for x1, y1, x2, y2 in group])


def get_group_average(group):
    x1 = int(np.mean([line[0] for line in group]))
    y1 = int(np.mean([line[1] for line in group]))
    x2 = int(np.mean([line[2] for line in group]))
    y2 = int(np.mean([line[3] for line in group]))
    return x1, y1, x2, y2


def get_best_lanes(groups):
    group_size_weight = 0.5
    sum_of_distances_from_opposite_lanes_weight = 0.5
    sum_of_lengths_weight = 0.5

    groups_scores_averages_left = []
    groups_scores_averages_right = []

    # Separate groups into left and right lanes based on angle
    for group in groups:
        x1, y1, x2, y2 = group[0]  # Assume first line in the group determines the angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if (
            consts.DEFAULT_LEFT_ANGLE_RANGE[0]
            <= angle
            <= consts.DEFAULT_LEFT_ANGLE_RANGE[1]
        ):
            groups_scores_averages_left.append(group)
        elif (
            consts.DEFAULT_RIGHT_ANGLE_RANGE[0]
            <= angle
            <= consts.DEFAULT_RIGHT_ANGLE_RANGE[1]
        ):
            groups_scores_averages_right.append(group)

    # Compute scores for left and right groups
    def compute_scores(groups):
        scored_groups = []
        for group in groups:
            size_score = group_size_weight * len(group)
            length_score = sum_of_lengths_weight * get_sum_of_lengths(group)
            scored_groups.append((group, size_score + length_score))
        return scored_groups

    scored_left = compute_scores(groups_scores_averages_left)
    scored_right = compute_scores(groups_scores_averages_right)

    # Compute distance-based scoring for left groups
    for i, (group, score) in enumerate(scored_left):
        x1l, _, x2l, _ = get_group_average(group)
        distance_score = 0
        for other_group, _ in scored_right:
            x1r, _, x2r, _ = get_group_average(other_group)
            distance_score += abs(x1l - x1r) + abs(x2l - x2r)
        scored_left[i] = (
            group,
            score + sum_of_distances_from_opposite_lanes_weight * distance_score,
        )

    # Compute distance-based scoring for right groups
    for i, (group, score) in enumerate(scored_right):
        x1r, _, x2r, _ = get_group_average(group)
        distance_score = 0
        for other_group, _ in scored_left:
            x1l, _, x2l, _ = get_group_average(other_group)
            distance_score += abs(x1l - x1r) + abs(x2l - x2r)
        scored_right[i] = (
            group,
            score + sum_of_distances_from_opposite_lanes_weight * distance_score,
        )

    # Select the best left and right lanes
    best_left_lane_group = max(scored_left, key=lambda x: x[1], default=(None, 0))[0]
    best_right_lane_group = max(scored_right, key=lambda x: x[1], default=(None, 0))[0]

    best_left_lane = (
        None
        if best_left_lane_group is None
        else get_group_average(best_left_lane_group)
    )
    best_right_lane = (
        None
        if best_right_lane_group is None
        else get_group_average(best_right_lane_group)
    )

    # Return the averaged best lanes
    return best_left_lane, best_right_lane


def filer_lines(lines):
    lines = remove_lines_by_angle(lines)
    groups = group_close_lines(lines, 90)
    best_left_lane, best_right_lane = get_best_lanes(groups)
    lanes = []
    if best_left_lane is not None:
        lanes.append(best_left_lane)
    if best_right_lane is not None:
        lanes.append(best_right_lane)

    return lanes


def extend_polygon_to_bottom(points, image_height):
    points = points.reshape(-1, 2)

    left_lane_point = points[0]
    right_lane_point = points[-1]

    left_slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0] + 1e-6)
    right_slope = (points[-1][1] - points[-2][1]) / (
        points[-1][0] - points[-2][0] + 1e-6
    )

    left_bottom_x = (
        left_lane_point[0] + (image_height - left_lane_point[1]) / left_slope
    )
    right_bottom_x = (
        right_lane_point[0] + (image_height - right_lane_point[1]) / right_slope
    )

    extended_polygon = np.array(
        [
            [left_lane_point[0], left_lane_point[1]],
            [points[1][0], points[1][1]],
            [points[-2][0], points[-2][1]],
            [right_lane_point[0], right_lane_point[1]],
            [right_bottom_x, image_height],
            [left_bottom_x, image_height],
        ],
        dtype=np.int32,
    )

    return extended_polygon


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
            lines = filer_lines(hough_linesp)
            if len(lines) == 2:
                x1, y1, x2, y2 = lines[0]
                x3, y3, x4, y4 = lines[1]
                # Apply inverse perspective transform to the line endpoints
                points = np.array(
                    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32
                ).reshape(-1, 1, 2)
                if (
                    not apply_perspective
                    and inverse_perspective_transform_matrix is not None
                    or base_layer
                ):
                    transformed_points = cv2.perspectiveTransform(
                        points, inverse_perspective_transform_matrix
                    )
                    poly_points = extend_polygon_to_bottom(
                        transformed_points, frame.shape[0]
                    )
                else:
                    transformed_points = points

                # fill the lane
                cv2.fillPoly(output_frame, np.int32([poly_points]), (0, 255, 0))
    else:
        output_frame = frame

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
