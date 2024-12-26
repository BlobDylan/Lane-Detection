import cv2
import numpy as np
import argparse
import consts
import sliders


def get_region_of_interest_mask(frame):
    mask = np.zeros_like(frame)
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = frame.shape[:2]

    bottom_left = [0, rows]
    middle_left = [0, rows * 0.85]
    top_left = [cols * 0.4, rows * 0.7]

    bottom_right = [cols, rows]
    middle_right = [cols, rows * 0.85]
    top_right = [cols * 0.6, rows * 0.7]

    vertices = np.array(
        [[bottom_left, middle_left, top_left, top_right, middle_right, bottom_right]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return mask


def detect_lanes_in_frame(
    frame,
    region_of_interest_mask=None,
    canny_low_th=100,
    canny_high_th=200,
    apply_region=1,
    apply_canny=1,
):
    # applying region of interest mask
    if apply_region and region_of_interest_mask is not None:
        frame = cv2.bitwise_and(frame, region_of_interest_mask)

    # applying Canny
    if apply_canny:
        frame = cv2.Canny(frame, canny_low_th, canny_high_th)

    # resizing the frame
    frame = cv2.resize(frame, (consts.DISPLAY_WIDTH, consts.DISPLAY_HEIGHT))
    return None, frame


def main(args):
    cap = cv2.VideoCapture(consts.VIDEO_PATH)
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

            # get the region of interest only once.
            if region_of_interest_mask is None:
                region_of_interest_mask = get_region_of_interest_mask(input_frame)

            # get the current trackbar values if in slider mode
            if args.sliders:
                apply_region, apply_canny, canny_low_th, canny_high_th = (
                    sliders_instance.get_values()
                )
            else:
                canny_low_th = consts.DEFAULT_CANNY_LOW_TH
                canny_high_th = consts.DEFAULT_CANNY_HIGH_TH
                apply_region = 1
                apply_canny = 1

            # detect lanes in the frame
            err, output_frame = detect_lanes_in_frame(
                input_frame,
                region_of_interest_mask,
                canny_low_th,
                canny_high_th,
                apply_region,
                apply_canny,
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
