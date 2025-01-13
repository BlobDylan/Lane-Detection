import consts
import numpy as np
import cv2
import argparse
import lanedetector
import cardetector


def mark_average_of_groups_in_frame(frame, averages_of_groups):
    for point in averages_of_groups:
        cv2.circle(frame, (int(point[1]), int(point[0])), 75, (0, 0, 255), -1)
    return frame


def mark_polygon_in_frame(frame, poly_points):
    cv2.fillPoly(frame, np.int32([poly_points]), (0, 255, 0))


def write_moving_lanes(frame, direction):
    cv2.putText(
        frame,
        f"MOVING LANES {direction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


def calculate_distance_to_car(car_point, H, camera_height=1.5):
    # Convert the car point to homogeneous coordinates
    car_point_h = np.array([car_point[1], car_point[0], 1], dtype=np.float32).reshape(
        3, 1
    )

    # Transform the point using the homography matrix
    world_point_h = np.dot(H, car_point_h)

    # Normalize to get real-world coordinates
    if world_point_h[2, 0] == 0:
        return None  # Avoid division by zero
    world_point = world_point_h[:2] / world_point_h[2]

    # Calculate real-world vertical distance (y_real)
    y_real = float(abs(world_point[1]))

    # Calculate 3D distance using Pythagoras
    distance = np.sqrt(y_real**2 + camera_height**2)

    return distance


def main(args):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)
        return

    frame_shape = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    lanedetector_instance = lanedetector.LaneDetector(frame_shape, args.sliders)
    cardetector_instance = cardetector.CarDetector()
    pause = False

    sample_lane_polygon = np.array([[925, 840], [1100, 840], [1600, 1080], [617, 1080]])
    lane_width = 3.5  # meters
    lane_length = 12  # meters

    real_world_points = np.array(
        [[0, 0], [lane_width, 0], [lane_width, lane_length], [0, lane_length]],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(sample_lane_polygon, real_world_points)

    try:
        while True:
            ret, input_frame = cap.read()
            if not ret:
                break

            # get the current trackbar values if in slider mode
            lanedetector_instance.get_sliders_values()

            # get the masks if needed avoiding repeated calculations
            lanedetector_instance.define_masks_if_needed()

            # detect lanes in the frame
            err, poly_points, is_moving_lanes, direction = (
                lanedetector_instance.detect_lanes_in_frame(frame=input_frame)
            )

            if is_moving_lanes:
                write_moving_lanes(input_frame, direction)

            if poly_points is not None:
                mark_polygon_in_frame(input_frame, poly_points)

            # handle errors detecting lanes
            if err:
                print(f"Error: {err}")
                break

            # detect cars in the frame
            averages_of_groups = cardetector_instance.detect_cars_in_frame(input_frame)

            output_frame = input_frame.copy()

            distances = []
            if poly_points is not None and H is not None:
                for car_point in averages_of_groups:
                    car_point = (
                        float(car_point[0]),
                        float(car_point[1]),
                    )  # Convert to tuple
                    distance = calculate_distance_to_car(car_point, H)
                    if distance is not None:
                        distances.append((car_point, distance))

            # Annotate frame with distances
            for car_point, distance in distances:
                cv2.putText(
                    output_frame,
                    f"{distance:.1f}m",
                    (int(car_point[1]), int(car_point[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # draw sample_lane_polygon
            cv2.polylines(
                output_frame,
                [np.int32(sample_lane_polygon)],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2,
            )

            # output_frame = mark_average_of_groups_in_frame(
            #     input_frame, averages_of_groups
            # )

            # get perspective transform matrix
            # resizing the frame just so it fits in the screen for display purposes.
            output_frame = cv2.resize(
                output_frame, (consts.DISPLAY_WIDTH, consts.DISPLAY_HEIGHT)
            )

            # show the frame
            cv2.imshow("Frame", output_frame)

            # break on 'q' key
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
            if cv2.waitKey(25) & 0xFF == ord("p"):
                pause = True
            while pause:
                if cv2.waitKey(25) & 0xFF == ord("u"):
                    pause = False
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection")
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file",
    )
    parser.add_argument(
        "--sliders",
        "-s",
        action="store_true",
        default=False,
        help="Show sliders to adjust parameters",
    )
    args = parser.parse_args()
    main(args)
