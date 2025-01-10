import cv2
import argparse
import lanedetector


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
            err, output_frame = lanedetector_instance.detect_lanes_in_frame(
                frame=input_frame
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
