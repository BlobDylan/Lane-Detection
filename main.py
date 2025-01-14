import consts
import numpy as np
import cv2
import argparse
import lanedetector
import cardetector


def draw_lines_between_tail_lights(frame, pairs_of_groups):
    for points in pairs_of_groups:
        pt1 = tuple(map(int, points[0]))
        pt2 = tuple(map(int, points[1]))
        cv2.line(frame, [pt2[1], pt2[0]], [pt1[1], pt1[0]], (0, 255, 0), 2)
        pixel_width = np.abs(pt1[1] - pt2[1])
        distance = calculate_distance(pixel_width + 100)
        cv2.putText(
            frame,
            f"{distance:.2f}m",
            (int((pt1[1] + pt2[1]) / 2), int((pt1[0] + pt2[0]) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
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


def calculate_distance(pixel_width):
    # Calculate widths in pixels
    top_width = 100
    bottom_width = 850

    # Normalize the input width between 0 and 1
    ratio = (bottom_width - pixel_width) / (bottom_width - top_width)

    # Convert to distance (linear interpolation)
    distance = 9 * ratio

    return distance


def get_average_frame_brightness(frame):
    return np.mean(frame)


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
    night_mode = False
    pause = False
    sample_lane_polygon = np.array([[950, 800], [1050, 800], [1450, 1080], [600, 1080]])

    try:
        while True:
            ret, input_frame = cap.read()
            if not ret:
                break

            night_mode = (
                get_average_frame_brightness(input_frame)
                < consts.DEFAULT_NIGHT_THRESHOLD
            )
            lanedetector_instance.set_night_mode(night_mode)
            cardetector_instance.set_night_mode(night_mode)

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
            pairs_of_groups = cardetector_instance.detect_cars_in_frame(input_frame)
            # frame = cardetector_instance.detect_cars_in_frame(input_frame)

            output_frame = input_frame.copy()
            # output_frame = frame

            output_frame = draw_lines_between_tail_lights(input_frame, pairs_of_groups)

            # draw sample_lane_polygon
            # cv2.polylines(
            #     output_frame,
            #     [np.int32(sample_lane_polygon)],
            #     isClosed=True,
            #     color=(0, 0, 255),
            #     thickness=2,
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
