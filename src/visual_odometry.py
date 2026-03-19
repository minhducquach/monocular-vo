import cv2
import numpy as np

from utils import init_pose, choose_next_frame, compute_relative_scale, visualize
from modules import Map

class VisualOdometry():
    def __init__(self, frames, camera, detector, matcher, ground_truth):
        self.frames = frames
        self.camera = camera
        self.feature_detector = detector
        self.feature_matcher = matcher
        self.map = Map()

        self.ground_truth_poses = ground_truth
        
        self.is_first_frame = True
        self.prev_frame = None
        
        self.keyframe_step = 2
        
        self.last_matches = None
        
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
            
            prev_keypoints = self.map.last_frame_keypoints()
            
            # CORRECTED:
            pts_prev = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
            pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])
            
            E, mask = cv2.findEssentialMat(np.float32(pts_curr), np.float32(pts_prev), self.camera.K)
            
            _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, self.camera.K, mask)
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            relative_scale = 0

            if self.last_matches == None:
                self.last_matches = matches
            
            else:
                relative_scale = compute_relative_scale(self.camera, self.map, self.last_matches, matches, keypoints, T)
                
            if relative_scale != 0:
                T[:3, 3] *= relative_scale
            
            curr_pose = self.map.last_frame_pose() @ T
            self.map.add_frame(curr_pose, keypoints, descriptors)  
            
            visualize(self.map, self.ground_truth_poses, self.prev_frame, frame, keypoints, descriptors, matches, id)
            
            self.prev_frame = frame
         
        # Viz img
        # cv2.imshow('img', frame)
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     return False
        # return True

        # Viz keypoints
        # keypoint_img = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=0)
        # cv2.imshow('img', keypoint_img)
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     return False
        # return True

    def run(self):
        for id, frame in enumerate(self.frames):
            self.process_frame(id, frame)
        #     if not self.process_frame(id, frame):
        #         break
        # cv2.destroyAllWindows()