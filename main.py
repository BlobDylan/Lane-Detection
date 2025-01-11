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

            # mark the averages of the groups in the frame
            output_frame = mark_average_of_groups_in_frame(
                input_frame, averages_of_groups
            )

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
