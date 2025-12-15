import cv2 as cv
import numpy as np

cap = cv.VideoCapture("video/test_video.mp4")

delay = 20  # usporenje (ms)

# Sliding window parametri
nwindows = 9
margin = 100
minpix = 50

out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(blur, 120, 255, cv.THRESH_BINARY)


    # --- KORAK 3: ROI ---
    h, w = binary.shape
    mask = np.zeros_like(binary)

    # trapez region ispred auta
    roi_vertices = np.array([[
        (int(0.2 * w), h),  # donji levi ugao
        (int(0.35 * w), int(0.6 * h)),  # gornji levi
        (int(0.6 * w), int(0.6 * h)),  # gornji desni
        (int(0.85 * w), h)  # donji desni
    ]], dtype=np.int32)

    cv.fillPoly(mask, roi_vertices, 255)
    roi = cv.bitwise_and(binary, mask)

    # Perspective transform
    src = np.float32([
        [0.35 * w, 0.6 * h],  # gornji levi
        [0.6 * w, 0.6 * h],  # gornji desni
        [0.85 * w, h],  # donji desni
        [0.2 * w, h]  # donji levi
    ])
    dst = np.float32([
        [0.25 * w, 0],  # gornji levi
        [0.75 * w, 0],  # gornji desni
        [0.75 * w, h],  # donji desni
        [0.25 * w, h]  # donji levi
    ])

    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(roi, M, (w, h))

    # ---------- KORAK 5: Sliding Window ----------
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = warped.shape[0] // nwindows
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    out_img = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)  # za crtanje

    for window in range(nwindows):
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Prikaz prozora
        cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 0, 255), 2)
        cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = None
    right_fit = None

    # Fit parabole
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    # Crtanje linija
    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        if 'left_fit' in locals():
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            for i in range(len(ploty) - 1):
                cv.line(out_img, (int(left_fitx[i]), int(ploty[i])),
                        (int(left_fitx[i + 1]), int(ploty[i + 1])),
                        (0, 255, 0), 3)
        if 'right_fit' in locals():
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            for i in range(len(ploty) - 1):
                cv.line(out_img, (int(right_fitx[i]), int(ploty[i])),
                        (int(right_fitx[i + 1]), int(ploty[i + 1])),
                        (0, 255, 0), 3)


    # ---------- KORAK 6: Overlay linija ----------
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    Minv = cv.getPerspectiveTransform(dst, src)
    newwarp = cv.warpPerspective(color_warp, Minv, (w, h))
    result = cv.addWeighted(frame, 1, newwarp, 0.3, 0)

    # ---------- Offset od centra ----------
    if left_fit is not None and right_fit is not None:
        car_position = w / 2
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        center_offset_pixels = car_position - lane_center
        xm_per_pix = 3.7/700
        center_offset_mtrs = center_offset_pixels * xm_per_pix
        cv.putText(result, f"Offset: {center_offset_mtrs:.2f} m", (50,50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # ---------- GRID PRIKAZ ----------
    binary_bgr = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    roi_bgr = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
    scale = 0.5
    orig_resized = cv.resize(result, (0,0), fx=scale, fy=scale)
    binary_resized = cv.resize(binary_bgr, (0,0), fx=scale, fy=scale)
    roi_resized = cv.resize(roi_bgr, (0,0), fx=scale, fy=scale)
    sliding_resized = cv.resize(out_img, (0,0), fx=scale, fy=scale)

    top_row = np.hstack((orig_resized, binary_resized))
    bottom_row = np.hstack((roi_resized, sliding_resized))
    grid = np.vstack((top_row, bottom_row))

    cv.putText(grid, "Original w/ overlay", (20, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.putText(grid, "Binary", (660, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv.putText(grid, "ROI", (20, 400),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.putText(grid, "Sliding", (660, 400),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(grid, "windows", (660, 440),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if out is None:
        h, w = grid.shape[:2]
        fps = cap.get(cv.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        out = cv.VideoWriter(
            "lane_pipeline_grid.mp4",
            cv.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )

    out.write(grid)

    cv.imshow("Pipeline Grid", grid)
    #cv.imshow("Final Overlay", result)


    key = cv.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()