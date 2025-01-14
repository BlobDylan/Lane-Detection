import cv2
import numpy as np
import consts


class CarDetector:
    def __init__(self):
        self.night_mode = False
        self.red_lower = np.array(consts.DEFAULT_RED_LOWER)
        self.red_upper = np.array(consts.DEFAULT_RED_UPPER)
        self.dilate_kernel_size = consts.DEFAULT_DILATE_KERNEL_SIZE_CARS
        self.car_threshold = consts.DEFAULT_CAR_THRESHOLD
        self.car_points_history_size = consts.DEFAULT_CAR_POINTS_HISTORY_SIZE
        self.pair_closeness_threshold = consts.DEFAULT_PAIR_CLOSENESS_THRESHOLD
        self.history_closeness_threshold = consts.DEFAULT_HISTORY_CLOSENESS_THRESHOLD
        self.kmeans_k = consts.DEFAULT_CARS_KMEANS_K
        self.points_history = []

    def set_night_mode(self, night_mode):
        self.night_mode = night_mode

    def apply_red_mask(self, frame):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for red
        mask = cv2.inRange(hsv_image, self.red_lower, self.red_upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        return res

    # use kmeans to group close points

    def group_close_points_in_frame(self, frame, distance_threshold=50):
        points = np.argwhere(frame == 255)
        points = np.float32(points)
        K = self.kmeans_k

        if len(points) <= K:
            return []

        # Perform k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(
            points, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Initialize groups based on the clustering
        initial_groups = [points[labels.ravel() == i] for i in range(K)]
        merged_groups = []

        for group in initial_groups:
            if len(merged_groups) == 0:
                merged_groups.append(group)
                continue

            # Check if this group should merge with an existing group
            merged = False
            for i, merged_group in enumerate(merged_groups):
                if (
                    np.linalg.norm(
                        np.mean(group, axis=0) - np.mean(merged_group, axis=0)
                    )
                    < distance_threshold
                ):
                    merged_groups[i] = np.vstack((merged_group, group))
                    merged = True
                    break

            # If no merging occurred, create a new group
            if not merged:
                merged_groups.append(group)

        return merged_groups

    def get_averages_of_groups(self, groups):
        averages_of_groups = []
        for group in groups:
            average = np.mean(group, axis=0)
            averages_of_groups.append(average)
        return averages_of_groups

    # given the averages of groups, return pairs of groups that are close to each other if under a certain threshold
    def get_pairs_of_groups(self, averages_of_groups, threshold):
        pairs = []
        for i in range(len(averages_of_groups)):
            for j in range(i + 1, len(averages_of_groups)):
                if (
                    np.linalg.norm(averages_of_groups[i] - averages_of_groups[j])
                    < threshold
                ):
                    pairs.append((averages_of_groups[j], averages_of_groups[i]))
        return pairs

    # only keep groups that have close points in the history
    def filter_groups(self, averages_of_groups):
        if len(self.points_history) == 0:
            return []

        filtered_groups = []
        for group in averages_of_groups:
            count = 0
            for points in self.points_history:
                has_close_point_in_history = False
                for point in points:
                    if np.linalg.norm(group - point) < self.history_closeness_threshold:
                        has_close_point_in_history = True
                        break
                if has_close_point_in_history:
                    count += 1
            if count == len(self.points_history):
                filtered_groups.append(group)
        return filtered_groups

    def detect_cars_in_frame(self, frame):
        output_frame = self.apply_red_mask(frame)

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)

        _, output_frame = cv2.threshold(
            output_frame,
            self.car_threshold[0],
            self.car_threshold[1],
            cv2.THRESH_BINARY,
        )

        output_frame = cv2.dilate(
            output_frame,
            np.ones(
                (self.dilate_kernel_size * 2 + 1, self.dilate_kernel_size * 2 + 1),
                np.uint8,
            ),
        )

        groups = self.group_close_points_in_frame(output_frame)
        averages_of_groups = self.get_averages_of_groups(groups)
        if len(averages_of_groups) > 0:
            self.points_history.append(averages_of_groups)
            if len(self.points_history) > self.car_points_history_size:
                self.points_history.pop(0)

        pairs = self.get_pairs_of_groups(
            averages_of_groups, self.pair_closeness_threshold
        )
        return pairs
        # return output_frame
