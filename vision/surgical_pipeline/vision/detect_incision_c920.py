#!/usr/bin/env python3

import cv2
import numpy as np


CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480

# ============================================================
# C920 고정 시점 기준 피부 ROI
# 화면에는 ROI 박스를 표시하지 않지만,
# 검출 자체는 이 영역 안에서만 수행
# ============================================================

ROI_X1 = 250
ROI_Y1 = 170
ROI_X2 = 385
ROI_Y2 = 355


# ============================================================
# 실제 화면에서 조정한 HSV 값
# ============================================================

H1_MAX = 9
H2_MIN = 136
S_MIN = 25
V_MIN = 61


# ============================================================
# 절개선 contour 조건
# ============================================================

MIN_AREA = 8
MIN_HEIGHT = 12
MAX_WIDTH = 35
MIN_VERTICAL_RATIO = 1.5


def nothing(x):
    pass


def find_incision(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if h < MIN_HEIGHT:
            continue

        if w > MAX_WIDTH:
            continue

        ratio = h / max(w, 1)

        if ratio < MIN_VERTICAL_RATIO:
            continue

        # 세로로 길수록 우선
        score = h * ratio

        candidates.append(
            (
                score,
                contour
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0][1]


def contour_endpoints(contour):

    pts = contour.reshape(
        -1,
        2
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    # 절개선 중심 X
    center_x = int(
        x + w / 2
    )

    # ========================================================
    # 화면 아래쪽 = START
    # 화면 위쪽   = END
    #
    # OpenCV:
    # 아래쪽일수록 y 값이 큼
    # ========================================================

    start_raw = pts[
        np.argmax(
            pts[:, 1]
        )
    ]

    end_raw = pts[
        np.argmin(
            pts[:, 1]
        )
    ]

    start = (
        center_x,
        int(start_raw[1])
    )

    end = (
        center_x,
        int(end_raw[1])
    )

    return start, end


def main():

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    if not cap.isOpened():

        print(
            "❌ C920 open failed"
        )

        return


    # ========================================================
    # HSV 조절창
    # ========================================================

    cv2.namedWindow(
        "HSV CONTROL",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "HSV CONTROL",
        500,
        300
    )

    cv2.createTrackbar(
        "H1 MAX",
        "HSV CONTROL",
        H1_MAX,
        30,
        nothing
    )

    cv2.createTrackbar(
        "H2 MIN",
        "HSV CONTROL",
        H2_MIN,
        179,
        nothing
    )

    cv2.createTrackbar(
        "S MIN",
        "HSV CONTROL",
        S_MIN,
        255,
        nothing
    )

    cv2.createTrackbar(
        "V MIN",
        "HSV CONTROL",
        V_MIN,
        255,
        nothing
    )


    print()
    print("========================================")
    print(" C920 INCISION HSV DETECTOR")
    print("========================================")
    print()
    print("q : 종료")
    print("s : 현재 START / END 출력")
    print()
    print("START = 화면 아래쪽")
    print("END   = 화면 위쪽")
    print()
    print("START → END = 아래 → 위")
    print()
    print(
        f"HSV DEFAULT : "
        f"H1_MAX={H1_MAX}, "
        f"H2_MIN={H2_MIN}, "
        f"S_MIN={S_MIN}, "
        f"V_MIN={V_MIN}"
    )
    print()
    print("========================================")


    last_start = None
    last_end = None

    previous_start = None
    previous_end = None

    # 좌표 떨림 완화
    ALPHA = 0.30


    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "❌ camera frame read failed"
            )

            break


        frame = cv2.resize(
            frame,
            (
                WIDTH,
                HEIGHT
            )
        )

        display = frame.copy()


        # ====================================================
        # ROI
        #
        # 검출은 이 영역에서만 하지만
        # 화면에 네모 박스는 그리지 않음
        # ====================================================

        roi = frame[
            ROI_Y1:ROI_Y2,
            ROI_X1:ROI_X2
        ]


        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )


        # ====================================================
        # 실시간 HSV Trackbar 값
        # ====================================================

        h1_max = cv2.getTrackbarPos(
            "H1 MAX",
            "HSV CONTROL"
        )

        h2_min = cv2.getTrackbarPos(
            "H2 MIN",
            "HSV CONTROL"
        )

        s_min = cv2.getTrackbarPos(
            "S MIN",
            "HSV CONTROL"
        )

        v_min = cv2.getTrackbarPos(
            "V MIN",
            "HSV CONTROL"
        )


        # ====================================================
        # RED MASK
        # ====================================================

        lower1 = np.array(
            [
                0,
                s_min,
                v_min
            ],
            dtype=np.uint8
        )

        upper1 = np.array(
            [
                h1_max,
                255,
                255
            ],
            dtype=np.uint8
        )


        lower2 = np.array(
            [
                h2_min,
                s_min,
                v_min
            ],
            dtype=np.uint8
        )

        upper2 = np.array(
            [
                179,
                255,
                255
            ],
            dtype=np.uint8
        )


        mask1 = cv2.inRange(
            hsv,
            lower1,
            upper1
        )

        mask2 = cv2.inRange(
            hsv,
            lower2,
            upper2
        )

        mask = cv2.bitwise_or(
            mask1,
            mask2
        )


        # ====================================================
        # Noise 제거
        # ====================================================

        kernel_open = np.ones(
            (
                2,
                2
            ),
            np.uint8
        )

        kernel_close = np.ones(
            (
                3,
                3
            ),
            np.uint8
        )


        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel_open
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel_close
        )


        contour = find_incision(
            mask
        )


        if contour is not None:

            start_roi, end_roi = (
                contour_endpoints(
                    contour
                )
            )


            # ROI 좌표 → 전체 영상 좌표

            sx = (
                start_roi[0]
                + ROI_X1
            )

            sy = (
                start_roi[1]
                + ROI_Y1
            )

            ex = (
                end_roi[0]
                + ROI_X1
            )

            ey = (
                end_roi[1]
                + ROI_Y1
            )


            # =================================================
            # 좌표 smoothing
            # =================================================

            if previous_start is not None:

                sx = int(
                    ALPHA * sx
                    +
                    (1 - ALPHA)
                    * previous_start[0]
                )

                sy = int(
                    ALPHA * sy
                    +
                    (1 - ALPHA)
                    * previous_start[1]
                )

                ex = int(
                    ALPHA * ex
                    +
                    (1 - ALPHA)
                    * previous_end[0]
                )

                ey = int(
                    ALPHA * ey
                    +
                    (1 - ALPHA)
                    * previous_end[1]
                )


            start = (
                sx,
                sy
            )

            end = (
                ex,
                ey
            )


            previous_start = start
            previous_end = end

            last_start = start
            last_end = end


            # contour를 전체 영상 좌표로 변환

            contour_global = (
                contour
                +
                np.array(
                    [
                        [
                            ROI_X1,
                            ROI_Y1
                        ]
                    ]
                )
            )


            # 절개선 contour
            cv2.drawContours(
                display,
                [
                    contour_global
                ],
                -1,
                (
                    0,
                    255,
                    255
                ),
                1
            )


            # =================================================
            # START → END 방향
            # =================================================

            cv2.arrowedLine(
                display,
                start,
                end,
                (
                    255,
                    0,
                    255
                ),
                2,
                tipLength=0.10
            )


            # =================================================
            # START
            # 기존보다 작은 점
            # =================================================

            cv2.circle(
                display,
                start,
                4,
                (
                    0,
                    255,
                    0
                ),
                -1
            )

            cv2.circle(
                display,
                start,
                7,
                (
                    0,
                    255,
                    0
                ),
                1
            )


            cv2.putText(
                display,
                f"START ({start[0]}, {start[1]})",
                (
                    start[0] + 9,
                    start[1] + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    0,
                    255,
                    0
                ),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # END
            # 기존보다 작은 점
            # =================================================

            cv2.circle(
                display,
                end,
                4,
                (
                    0,
                    0,
                    255
                ),
                -1
            )

            cv2.circle(
                display,
                end,
                7,
                (
                    0,
                    0,
                    255
                ),
                1
            )


            cv2.putText(
                display,
                f"END ({end[0]}, {end[1]})",
                (
                    end[0] + 9,
                    end[1] - 9
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    0,
                    0,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                display,
                "INCISION DETECTED",
                (
                    15,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (
                    0,
                    255,
                    0
                ),
                2,
                cv2.LINE_AA
            )


        else:

            cv2.putText(
                display,
                "INCISION NOT DETECTED",
                (
                    15,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (
                    0,
                    0,
                    255
                ),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # 중앙 십자선
        # ====================================================

        cv2.line(
            display,
            (
                WIDTH // 2,
                0
            ),
            (
                WIDTH // 2,
                HEIGHT
            ),
            (
                255,
                255,
                255
            ),
            1
        )

        cv2.line(
            display,
            (
                0,
                HEIGHT // 2
            ),
            (
                WIDTH,
                HEIGHT // 2
            ),
            (
                255,
                255,
                255
            ),
            1
        )


        cv2.imshow(
            "C920 - HSV INCISION START END",
            display
        )

        cv2.imshow(
            "INCISION MASK",
            mask
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):

            break


        elif key == ord("s"):

            if (
                last_start is not None
                and
                last_end is not None
            ):

                print()
                print(
                    "=============================="
                )

                print(
                    f"START_PIXEL = {last_start}"
                )

                print(
                    f"END_PIXEL   = {last_end}"
                )

                print(
                    "DIRECTION   = START -> END"
                )

                print(
                    "             BOTTOM -> TOP"
                )

                print(
                    "=============================="
                )

            else:

                print(
                    "❌ incision not detected"
                )


    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
