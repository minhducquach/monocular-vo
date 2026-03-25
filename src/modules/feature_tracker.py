import cv2
import numpy as np

class FeatureTracker():
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def track(self, prev_frame, curr_frame, prev_pts):
        """
        Tracks feature points from the previous frame to the current frame using Lucas-Kanade optical flow.
        """
        # prev_pts = [prev_pts[i].pt for i in range(len(prev_pts))]
        prev_pts = np.array(prev_pts, dtype=np.float32)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_pts, None, **self.lk_params)

        # Filter points based on status
        status = status.reshape(-1).astype(bool)
        valid_prev_pts = prev_pts[status]
        valid_curr_pts = curr_pts[status]

        return valid_prev_pts, valid_curr_pts
