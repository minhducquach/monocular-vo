import cv2
import matplotlib.pyplot as plt

from utils import init_pose, choose_next_frame
from modules import Map

class VisualOdometry():
    def __init__(self, frames, camera, detector, matcher):
        self.frames = frames
        self.camera = camera
        self.feature_detector = detector
        self.feature_matcher = matcher
        self.map = Map()
        
        self.is_first_frame = True
        self.prev_frame = None
        
        self.keyframe_step = 3
        
    def process_frame(self, id, frame):
        if self.is_first_frame:
            self.is_first_frame = False
            keypoints, descriptors = self.feature_detector.detect(frame)
            initial_pose = init_pose()
            self.map.add_frame(initial_pose, keypoints, descriptors)
            self.prev_frame = frame
            return True
        
        elif choose_next_frame(id, self.keyframe_step):
            keypoints, descriptors = self.feature_detector.detect(frame)
            matches = self.feature_matcher.match(self.map.last_frame_descriptors(), descriptors)
            
            # Viz matches
            match_img = cv2.drawMatches(self.prev_frame, self.map.last_frame_keypoints(), frame, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.imshow('img', match_img)
            
            self.prev_frame = frame
         
        # Viz img
        # cv2.imshow('img', frame)
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     return False
        # return True

        # Viz keypoints
        # keypoint_img = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=0)
        # cv2.imshow('img', keypoint_img)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            return False
        return True

    def run(self):
        for id, frame in enumerate(self.frames):
            # self.process_frame(frame)
            if not self.process_frame(id, frame):
                break
        cv2.destroyAllWindows()