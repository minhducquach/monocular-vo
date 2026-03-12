import cv2
import matplotlib.pyplot as plt

class VisualOdometry():
    def __init__(self, frames, camera, detector, matcher, map):
        self.frames = frames
        self.camera = camera
        self.feature_detector = detector
        self.feature_matcher = matcher
        self.map = map
        
        self.is_first_frame = True
        self.prev_frame = None
        
    def process_frame(self, frame):
        keypoints, descriptors = self.feature_detector.detect(frame)
        if self.is_first_frame:
            self.is_first_frame = False
            return True
        
        # Viz img
        # cv2.imshow('img', frame)
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     return False
        # return True

        # Viz keypoints
        keypoint_img = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=0)
        cv2.imshow('img', keypoint_img)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            return False
        return True

    def run(self):
        for frame in self.frames:
            # self.process_frame(frame)
            if not self.process_frame(frame):
                break
        cv2.destroyAllWindows()