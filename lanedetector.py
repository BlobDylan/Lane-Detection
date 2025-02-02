import cv2
import numpy as np
import sliders
import consts


class LaneDetector:
    def __init__(self, frame_shape, use_sliders=False):
        self.night_mode = False
        self.frame_shape = frame_shape
        self.use_sliders = use_sliders
        self.prev_gray = 1

        self.region_of_interest_mask = None
        self.perspective_transform_matrix = None
        self.inverse_perspective_transform_matrix = None

        self.base_layer = 1
        self.apply_color = 1
        self.apply_gray = 1
        self.apply_threshold = 1
        self.apply_region = 1
        self.apply_perspective = 1
        self.apply_canny = 1
        self.apply_hough = 1
        self.apply_filter_lines = 1
        self.gaussian_blur_kernel_size = consts.DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE
        self.canny_low_th = consts.DEFAULT_CANNY_LOW_TH
        self.canny_high_th = consts.DEFAULT_CANNY_HIGH_TH
        self.dilate_kernel_size = consts.DEFAULT_DILATE_KERNEL_SIZE
        self.hough_th = consts.DEFAULT_HOUGH_TH
        self.hough_min_line_length = consts.DEFAULT_HOUGH_MIN_LINE_LENGTH
        self.hough_max_line_gap = consts.DEFAULT_HOUGH_MAX_LINE_GAP
        self.yellow_lower = np.array(consts.DEFAULT_YELLOW_LOWER)
        self.yellow_upper = np.array(consts.DEFAULT_YELLOW_UPPER)
        self.white_lower = np.array(consts.DEFAULT_WHITE_LOWER)
        self.white_upper = np.array(consts.DEFAULT_WHITE_UPPER)
        self.left_lane_angle_range = consts.DEFAULT_LEFT_ANGLE_RANGE
        self.right_lane_angle_range = consts.DEFAULT_RIGHT_ANGLE_RANGE
        self.lane_history_size = consts.DEFAULT_LANE_HISTORY_SIZE
        self.intersection_x_history_size = consts.DEFAULT_INTERSECTION_X_HISTORY_SIZE
        self.delta_moving_lanes_threshold = consts.DEFAULT_DELTA_MOVING_LANES_THRESHOLD
        self.perspective_src = consts.DEFAULT_PERSPECTIVE_SRC

        self.lane_history = []
        self.consecutive_frames_no_lanes = 0
        self.distance_from_median_threshold = (
            consts.DEFAULT_DISTANCE_FROM_MEDIAN_THRESHOLD
        )

        self.is_moving_lanes = False
        self.lane_move_direction = None
        self.intersection_x_history_left = []
        self.intersection_x_history_right = []
        self.average_change_history = []

        if use_sliders:
            self.sliders_instance = sliders.Sliders()
            self.get_sliders_values()

    def set_night_mode(self, night_mode):
        self.night_mode = night_mode

        self.gaussian_blur_kernel_size = (
            consts.DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE_NIGHT
            if night_mode
            else consts.DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE
        )
        self.canny_low_th = (
            consts.DEFAULT_CANNY_LOW_TH_NIGHT
            if night_mode
            else consts.DEFAULT_CANNY_LOW_TH
        )
        self.canny_high_th = (
            consts.DEFAULT_CANNY_HIGH_TH_NIGHT
            if night_mode
            else consts.DEFAULT_CANNY_HIGH_TH
        )
        self.dilate_kernel_size = (
            consts.DEFAULT_DILATE_KERNEL_SIZE_NIGHT
            if night_mode
            else consts.DEFAULT_DILATE_KERNEL_SIZE
        )
        self.hough_th = (
            consts.DEFAULT_HOUGH_TH_NIGHT if night_mode else consts.DEFAULT_HOUGH_TH
        )
        self.hough_min_line_length = (
            consts.DEFAULT_HOUGH_MIN_LINE_LENGTH_NIGHT
            if night_mode
            else consts.DEFAULT_HOUGH_MIN_LINE_LENGTH
        )
        self.hough_max_line_gap = (
            consts.DEFAULT_HOUGH_MAX_LINE_GAP_NIGHT
            if night_mode
            else consts.DEFAULT_HOUGH_MAX_LINE_GAP
        )
        self.yellow_lower = (
            np.array(consts.DEFAULT_YELLOW_LOWER_NIGHT)
            if night_mode
            else np.array(consts.DEFAULT_YELLOW_LOWER)
        )
        self.yellow_upper = (
            np.array(consts.DEFAULT_YELLOW_UPPER_NIGHT)
            if night_mode
            else np.array(consts.DEFAULT_YELLOW_UPPER)
        )
        self.white_lower = (
            np.array(consts.DEFAULT_WHITE_LOWER_NIGHT)
            if night_mode
            else np.array(consts.DEFAULT_WHITE_LOWER)
        )
        self.white_upper = (
            np.array(consts.DEFAULT_WHITE_UPPER_NIGHT)
            if night_mode
            else np.array(consts.DEFAULT_WHITE_UPPER)
        )
        self.left_lane_angle_range = (
            consts.DEFAULT_LEFT_ANGLE_RANGE_NIGHT
            if night_mode
            else consts.DEFAULT_LEFT_ANGLE_RANGE
        )
        self.right_lane_angle_range = (
            consts.DEFAULT_RIGHT_ANGLE_RANGE_NIGHT
            if night_mode
            else consts.DEFAULT_RIGHT_ANGLE_RANGE
        )
        self.lane_history_size = (
            consts.DEFAULT_LANE_HISTORY_SIZE_NIGHT
            if night_mode
            else consts.DEFAULT_LANE_HISTORY_SIZE
        )
        self.intersection_x_history_size = (
            consts.DEFAULT_INTERSECTION_X_HISTORY_SIZE_NIGHT
            if night_mode
            else consts.DEFAULT_INTERSECTION_X_HISTORY_SIZE
        )
        self.delta_moving_lanes_threshold = (
            consts.DEFAULT_DELTA_MOVING_LANES_THRESHOLD_NIGHT
            if night_mode
            else consts.DEFAULT_DELTA_MOVING_LANES_THRESHOLD
        )
        self.perspective_src = (
            consts.DEFAULT_PERSPECTIVE_SRC_NIGHT
            if night_mode
            else consts.DEFAULT_PERSPECTIVE_SRC
        )
        self.distance_from_median_threshold = (
            consts.DEFAULT_DISTANCE_FROM_MEDIAN_THRESHOLD_NIGHT
            if night_mode
            else consts.DEFAULT_DISTANCE_FROM_MEDIAN_THRESHOLD
        )

    def get_sliders_values(self):
        if self.use_sliders:
            (
                self.base_layer,
                self.apply_color,
                self.apply_gray,
                self.apply_threshold,
                self.apply_region,
                self.apply_perspective,
                self.apply_canny,
                self.apply_hough,
                self.apply_filter_lines,
                self.gaussian_blur_kernel_size,
                self.canny_low_th,
                self.canny_high_th,
                self.dilate_kernel_size,
                self.hough_th,
                self.hough_min_line_length,
                self.hough_max_line_gap,
            ) = self.sliders_instance.get_values()

    def define_masks_if_needed(self):
        if self.apply_gray != self.prev_gray:
            self.region_of_interest_mask = None
            self.prev_gray = self.apply_gray

        if self.region_of_interest_mask is None:
            self.get_region_of_interest_mask()

        if self.perspective_transform_matrix is None:
            self.get_perspective_transform_matrix()

    def get_perspective_transform_matrix(self):
        rows, cols = self.frame_shape[:2]
        src = np.float32(
            [
                [
                    cols * self.perspective_src[0],
                    rows * self.perspective_src[1],
                ],  # 0.3, 0.75 top left
                [
                    cols * self.perspective_src[2],
                    rows * self.perspective_src[3],
                ],  # 0.7, 0.75 top right
                [
                    cols * self.perspective_src[4],
                    rows * self.perspective_src[5],
                ],  # 1, 1 btm right
                [
                    cols * self.perspective_src[6],
                    rows * self.perspective_src[7],
                ],  # 0, 1 btm left
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
        self.perspective_transform_matrix = cv2.getPerspectiveTransform(src, dst)
        self.inverse_perspective_transform_matrix = cv2.getPerspectiveTransform(
            dst, src
        )

    def get_region_of_interest_mask(self):
        rows, cols = self.frame_shape[:2]
        mask_shape = (rows, cols) if self.apply_gray else (rows, cols, 3)
        mask = np.zeros(mask_shape, dtype=np.uint8)

        if self.apply_gray:
            ignore_mask_color = 255
        else:
            channel_count = 1 if self.apply_gray else 3
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
        self.region_of_interest_mask = mask

    def get_color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(
            hsv,
            np.array(self.yellow_lower),
            np.array(self.yellow_upper),
        )
        mask_white = cv2.inRange(
            hsv,
            np.array(self.white_lower),
            np.array(self.white_upper),
        )

        mask = cv2.bitwise_or(mask_yellow, mask_white)
        return mask

    def remove_lines_by_angle(self, lines):
        filtered_lines = []

        left_lane_angle_range = self.left_lane_angle_range
        right_lane_angle_range = self.right_lane_angle_range

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 != x1:  # Avoid division by zero
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if (
                        left_lane_angle_range[0] <= angle <= left_lane_angle_range[1]
                        or right_lane_angle_range[0]
                        <= angle
                        <= right_lane_angle_range[1]
                    ):
                        filtered_lines.append((x1, y1, x2, y2))
        return filtered_lines

    def group_close_lines(self, lines, distance_threshold):
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

    def get_sum_of_lengths(self, group):
        return sum(
            [np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for x1, y1, x2, y2 in group]
        )

    def get_group_average(self, group):
        x1 = int(np.mean([line[0] for line in group]))
        y1 = int(np.mean([line[1] for line in group]))
        x2 = int(np.mean([line[2] for line in group]))
        y2 = int(np.mean([line[3] for line in group]))
        return x1, y1, x2, y2

    def get_best_lanes(self, groups):
        group_size_weight = 0.02
        sum_of_distances_from_opposite_lanes_weight = 0.9
        sum_of_lengths_weight = 0.08

        groups_scores_averages_left = []
        groups_scores_averages_right = []

        # Separate groups into left and right lanes based on angle
        for group in groups:
            x1, y1, x2, y2 = group[
                0
            ]  # Assume first line in the group determines the angle

            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            bottom_x = x1 + (self.frame_shape[0] - y1) / slope

            if bottom_x < self.frame_shape[1] / 2:
                groups_scores_averages_left.append(group)
            else:
                groups_scores_averages_right.append(group)

        # Compute scores for left and right groups
        def compute_scores(groups):
            scored_groups = []
            for group in groups:
                size_score = group_size_weight * len(group)
                length_score = sum_of_lengths_weight * self.get_sum_of_lengths(group)
                scored_groups.append((group, size_score + length_score))
            return scored_groups

        scored_left = compute_scores(groups_scores_averages_left)
        scored_right = compute_scores(groups_scores_averages_right)

        # Compute distance-based scoring for left groups
        for i, (group, score) in enumerate(scored_left):
            x1l, _, x2l, _ = self.get_group_average(group)
            distance_score = 0
            for other_group, _ in scored_right:
                x1r, _, x2r, _ = self.get_group_average(other_group)
                distance_score += abs(x1l - x1r) + abs(x2l - x2r)
            scored_left[i] = (
                group,
                score + sum_of_distances_from_opposite_lanes_weight * distance_score,
            )

        # Compute distance-based scoring for right groups
        for i, (group, score) in enumerate(scored_right):
            x1r, _, x2r, _ = self.get_group_average(group)
            distance_score = 0
            for other_group, _ in scored_left:
                x1l, _, x2l, _ = self.get_group_average(other_group)
                distance_score += abs(x1l - x1r) + abs(x2l - x2r)
            scored_right[i] = (
                group,
                score + sum_of_distances_from_opposite_lanes_weight * distance_score,
            )

        # Select the best left and right lanes
        best_left_lane_group = max(scored_left, key=lambda x: x[1], default=(None, 0))[
            0
        ]
        best_right_lane_group = max(
            scored_right, key=lambda x: x[1], default=(None, 0)
        )[0]

        best_left_lane = (
            None
            if best_left_lane_group is None
            else self.get_group_average(best_left_lane_group)
        )
        best_right_lane = (
            None
            if best_right_lane_group is None
            else self.get_group_average(best_right_lane_group)
        )

        # Return the averaged best lanes
        return best_left_lane, best_right_lane

    def filer_lines(self, lines):
        lines = self.remove_lines_by_angle(lines)
        groups = self.group_close_lines(lines, 80)
        best_left_lane, best_right_lane = self.get_best_lanes(groups)
        lanes = []
        if best_left_lane is not None:
            lanes.append(best_left_lane)
        if best_right_lane is not None:
            lanes.append(best_right_lane)

        return lanes

    def extend_polygon_to_bottom(self, points, image_height):
        points = points.reshape(-1, 2)

        left_lane_point = points[0]
        right_lane_point = points[-1]

        left_slope = (points[1][1] - points[0][1]) / (
            points[1][0] - points[0][0] + 1e-6
        )
        right_slope = (points[-1][1] - points[-2][1]) / (
            points[-1][0] - points[-2][0] + 1e-6
        )

        left_bottom_x = (
            left_lane_point[0] + (image_height - left_lane_point[1]) / left_slope + 1e-6
        )
        right_bottom_x = (
            right_lane_point[0]
            + (image_height - right_lane_point[1]) / right_slope
            + 1e-6
        )

        # avoid convering float inf to int
        if right_bottom_x == float("inf"):
            right_bottom_x = image_height
        if left_bottom_x == float("inf"):
            left_bottom_x = image_height

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
        self,
        frame,
    ):
        # applying color mask
        if self.apply_color:
            color_mask = self.get_color_mask(frame)
            frame = cv2.bitwise_and(frame, frame, mask=color_mask)

        # applying GaussianBlur
        frame = cv2.GaussianBlur(
            frame,
            (
                2 * self.gaussian_blur_kernel_size + 1,
                2 * self.gaussian_blur_kernel_size + 1,
            ),
            0,
        )

        # converting to grayscale
        if self.apply_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # applying threshold
        if self.apply_threshold:
            _, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

        # applying perspective transform
        if self.apply_perspective and self.perspective_transform_matrix is not None:
            frame = cv2.warpPerspective(
                frame,
                self.perspective_transform_matrix,
                (frame.shape[1], frame.shape[0]),
            )

        # applying region of interest mask
        if (
            not self.night_mode
            and self.apply_region
            and self.region_of_interest_mask is not None
        ):
            frame = cv2.bitwise_and(frame, self.region_of_interest_mask)

        # applying Canny
        if self.apply_canny:
            frame = cv2.Canny(frame, self.canny_low_th, self.canny_high_th)

        # Dilate the image to get a better result
        kernel = np.ones(
            (self.dilate_kernel_size * 2 + 1, self.dilate_kernel_size * 2 + 1), np.uint8
        )
        frame = cv2.dilate(frame, kernel, iterations=1)

        # applying HoughLines
        if self.apply_hough and not self.apply_canny and not self.apply_gray:
            return "HoughLines can only be applied after Canny or Gray", None

        if self.apply_hough:
            hough_linesp = cv2.HoughLinesP(
                frame,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_th,
                minLineLength=self.hough_min_line_length,
                maxLineGap=self.hough_max_line_gap,
            )
            poly_points = None
            if hough_linesp is not None:
                lines = self.filer_lines(hough_linesp)
                if len(lines) == 2:
                    median_lane = self.get_median_lane()
                    if median_lane is None or self.is_close_to_median(
                        lines, median_lane
                    ):
                        self.update_lane_history(lines)
                        x1, y1, x2, y2 = lines[0]
                        x3, y3, x4, y4 = lines[1]
                    else:
                        x1, y1, x2, y2 = median_lane[0]
                        x3, y3, x4, y4 = median_lane[1]
                    self.update_intersection_x_history_left(
                        self.get_line_intersection_x(x1, y1, x2, y2)
                    )
                    self.update_intersection_x_history_right(
                        self.get_line_intersection_x(x3, y3, x4, y4)
                    )
                    # Apply inverse perspective transform to the line endpoints
                    points = np.array(
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32
                    ).reshape(-1, 1, 2)
                    transformed_points = points
                    if (
                        self.base_layer
                        and self.apply_perspective
                        and self.inverse_perspective_transform_matrix is not None
                    ):
                        transformed_points = cv2.perspectiveTransform(
                            points, self.inverse_perspective_transform_matrix
                        )
                    poly_points = self.extend_polygon_to_bottom(
                        transformed_points, frame.shape[0]
                    )
                else:
                    self.consecutive_frames_no_lanes += 1
                    if self.consecutive_frames_no_lanes > 8:
                        self.lane_history = []
                        self.consecutive_frames_no_lanes = 0
                    # Draw the lines
                    if (
                        self.apply_perspective
                        and self.inverse_perspective_transform_matrix is not None
                    ):
                        for line in lines:
                            x1, y1, x2, y2 = line
                            if len(lines) == 1:
                                if self.is_closer_to_left(x1, y1, x2, y2):
                                    self.update_intersection_x_history_left(
                                        self.get_line_intersection_x(x1, y1, x2, y2)
                                    )
                                else:
                                    self.update_intersection_x_history_right(
                                        self.get_line_intersection_x(x1, y1, x2, y2)
                                    )
                self.update_moving_lanes()
        else:
            return

        return None, poly_points, self.is_moving_lanes, self.lane_move_direction

    def update_lane_history(self, lanes):
        self.lane_history.append(lanes)
        if len(self.lane_history) > self.lane_history_size:
            self.lane_history.pop(0)

    def get_median_lane(self):
        lanes = []
        for lane in self.lane_history:
            if len(lane) == 2:
                lanes.append(lane)
        if len(lanes) == 0:
            return None
        lanes = np.array(lanes)
        return np.median(lanes, axis=0)

    # closesness by IOU
    def is_close_to_median(self, lane, median_lane):
        if median_lane is None:
            return False
        if len(self.lane_history) < self.lane_history_size // 2:
            return True
        lane = np.array(lane)
        intersection = np.minimum(lane, median_lane)
        union = np.maximum(lane, median_lane)
        iou = np.sum(intersection) / np.sum(union) + 1e-6
        return iou > self.distance_from_median_threshold

    def update_intersection_x_history_left(self, x):
        self.intersection_x_history_left.append(x)
        if len(self.intersection_x_history_left) > self.intersection_x_history_size:
            self.intersection_x_history_left.pop(0)

    def update_intersection_x_history_right(self, x):
        self.intersection_x_history_right.append(x)
        if len(self.intersection_x_history_right) > self.intersection_x_history_size:
            self.intersection_x_history_right.pop(0)

    def get_line_intersection_x(self, x1, y1, x2, y2):
        y = self.frame_shape[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        return x1 + (y - y1) / (slope + 1e-6)

    def is_closer_to_left(self, x1, y1, x2, y2):
        if len(self.intersection_x_history_left) == 0:
            return True if self.lane_move_direction == "LEFT" else False
        intersection = self.get_line_intersection_x(x1, y1, x2, y2)
        left = np.array(self.intersection_x_history_left)
        right = np.array(self.intersection_x_history_right)
        left = np.sort(left)
        right = np.sort(right)
        left = left[3:-3]
        right = right[3:-3]
        median_left = np.median(left)
        median_right = np.median(right)
        return abs(intersection - median_left) < abs(intersection - median_right)

    def update_moving_lanes(self):
        if len(self.intersection_x_history_left) != self.intersection_x_history_size:
            return
        if len(self.intersection_x_history_right) != self.intersection_x_history_size:
            return
        # remove two largest and smallest values
        left = np.array(self.intersection_x_history_left)
        right = np.array(self.intersection_x_history_right)
        left_sorted = np.sort(left)
        right_sorted = np.sort(right)
        left_sorted = left_sorted[3:-3]
        right_sorted = right_sorted[3:-3]
        # keep original ordering but remove outliers
        left_without_outliers = np.array([x for x in left if x in left_sorted])
        right_without_outliers = np.array([x for x in right if x in right_sorted])

        if self.consecutive_frames_no_lanes > 0:
            if self.consecutive_frames_no_lanes > 5:
                self.intersection_x_history_left = []
                self.intersection_x_history_right = []
                self.average_change_history = []
                self.is_moving_lanes = False
                return
            if (
                abs(left_sorted[0] - left_sorted[-1])
                > self.delta_moving_lanes_threshold
                and abs(right_sorted[0] - right_sorted[-1])
                > self.delta_moving_lanes_threshold
            ):
                if (
                    left_without_outliers[-1] > left_without_outliers[0]
                    and right_without_outliers[-1] > right_without_outliers[0]
                ):
                    self.lane_move_direction = "LEFT"
                elif (
                    left_without_outliers[-1] < left_without_outliers[0]
                    and right_without_outliers[-1] < right_without_outliers[0]
                ):
                    self.lane_move_direction = "RIGHT"

                self.is_moving_lanes = True
                return
            self.is_moving_lanes = False
            return
        if (
            abs(left_sorted[0] - left_sorted[-1]) > self.delta_moving_lanes_threshold
            and abs(right_sorted[0] - right_sorted[-1])
            > self.delta_moving_lanes_threshold
        ):
            if (
                left_without_outliers[-1] > left_without_outliers[0]
                and right_without_outliers[-1] > right_without_outliers[0]
            ):
                self.lane_move_direction = "LEFT"
            elif (
                left_without_outliers[-1] < left_without_outliers[0]
                and right_without_outliers[-1] < right_without_outliers[0]
            ):
                self.lane_move_direction = "RIGHT"

            self.is_moving_lanes = True
            return
        self.is_moving_lanes = False
        return
