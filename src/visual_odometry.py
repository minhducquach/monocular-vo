import cv2
import numpy as np

from utils import init_pose, choose_next_frame, compute_relative_scale, compute_absolute_scale, visualize_matches, visualize_tracking, motion_check

from modules import Map

class VisualOdometry():
    def __init__(self, camera, detector, matcher, tracker, ground_truth, feature_mode, estimation_mode):
        # self.frames = frames
        self.camera = camera
        self.feature_detector = detector
        self.feature_matcher = matcher
        self.feature_tracker = tracker
        self.map = Map()

        self.ground_truth_poses = ground_truth
        
        self.is_first_frame = True
        self.prev_frame = None
        
        self.keyframe_step = 1
        
        self.bootstrap_frame = 2
        self.feature_mode = feature_mode
        
        self.num_min_features = 2000
        
        self.estimation_mode = estimation_mode
        
        # if self.feature_mode == 'matcher':
        #     self.last_matches = None
            
    def process_frame(self, id, frame):
        if self.is_first_frame:
            self.is_first_frame = False
            keypoints, descriptors = self.feature_detector.detect(frame)
            if self.feature_mode == 'tracker':
                keypoints = [keypoints[i].pt for i in range(len(keypoints))]
            initial_pose = init_pose()
            self.map.add_frame(initial_pose, keypoints, descriptors)
            self.prev_frame = frame
            return
        
        if id == self.bootstrap_frame:
            keypoints, descriptors = None, None
            prev_keypoints = self.map.last_frame_keypoints()
            
            pts_prev, pts_curr = None, None
            matches = None
            
            if self.feature_mode == 'matcher':
                keypoints, descriptors = self.feature_detector.detect(frame)
                # keypoints = [keypoints[i].pt for i in range(len(keypoints))]
                matches = self.feature_matcher.match(self.map.last_frame_descriptors(), descriptors)
                pts_prev = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])
            else:
                pts_prev, pts_curr = self.feature_tracker.track(self.prev_frame, frame, prev_keypoints) 
            
            E, mask = cv2.findEssentialMat(np.float32(pts_prev), np.float32(pts_curr), self.camera.K, cv2.RANSAC, 0.999, 1.0)
            
            print(mask.shape)
            # Pose from prev frame to curr frame (World to cam)
            _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.camera.K, mask)
            
            pts_curr_inliers = pts_curr[mask.ravel() == 1]
            pts_prev_inliers = pts_prev[mask.ravel() == 1]
            
            T = np.eye(4)
            
            if motion_check(pts_curr, pts_prev, threshold=3):
                
                # Inverse World to cam into Cam to world for pose accumulation
                T[:3, :3] = R.T
                T[:3, 3] = -R.T @ t.flatten()

                # relative_scale = 0

                # if self.last_matches == None:
                #     self.last_matches = matches
                
                # else:
                #     relative_scale = compute_relative_scale(self.camera, self.map, self.last_matches, matches, keypoints, T)
                    
                # if relative_scale != 0:
                #     T[:3, 3] *= relative_scale
            
                scale = compute_absolute_scale(self.ground_truth_poses, id, self.keyframe_step)
                if scale > 0.1:
                    T[:3, 3] *= scale
                    
            curr_pose = self.map.last_frame_pose() @ T
            
            P0 = self.camera.K @ self.map.last_frame_pose()[:3, :4]
            P1 = self.camera.K @ (curr_pose)[:3, :4]
            
            p = cv2.triangulatePoints(P0, P1, pts_prev_inliers.T, pts_curr_inliers.T)
            p = p / p[3]
            
            # Extract the 3D coordinates (x, y, z) and transpose (N, 3)
            landmarks = p[:3, :].T

            self.map.add_landmarks(landmarks)
            
            self.map.add_frame(curr_pose, pts_curr)
            self.prev_frame = frame
        
        # elif choose_next_frame(id, self.keyframe_step):
        elif id > self.bootstrap_frame:
            keypoints, descriptors = None, None
            prev_keypoints = self.map.last_frame_keypoints()
            
            pts_prev, pts_curr = None, None
            matches = None
            
            if self.feature_mode == 'matcher':
                keypoints, descriptors = self.feature_detector.detect(frame)
                # keypoints = [keypoints[i].pt for i in range(len(keypoints))]
                matches = self.feature_matcher.match(self.map.last_frame_descriptors(), descriptors)
                pts_prev = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])
            else:
                pts_prev, pts_curr = self.feature_tracker.track(self.prev_frame, frame, prev_keypoints) 
            
            E, mask = cv2.findEssentialMat(np.float32(pts_prev), np.float32(pts_curr), self.camera.K, cv2.RANSAC, 0.999, 1.0)
            
            # Pose from prev frame to curr frame (World to cam)
            _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.camera.K, mask)
            
            T = np.eye(4)
            
            if motion_check(pts_curr, pts_prev, threshold=3):
                
                # Inverse World to cam into Cam to world for pose accumulation
                T[:3, :3] = R.T
                T[:3, 3] = -R.T @ t.flatten()

                # relative_scale = 0

                # if self.last_matches == None:
                #     self.last_matches = matches
                
                # else:
                #     relative_scale = compute_relative_scale(self.camera, self.map, self.last_matches, matches, keypoints, T)
                    
                # if relative_scale != 0:
                #     T[:3, 3] *= relative_scale
            
                scale = compute_absolute_scale(self.ground_truth_poses, id, self.keyframe_step)
                if scale > 0.1:
                    T[:3, 3] *= scale
                       
            curr_pose = self.map.last_frame_pose() @ T
            if self.feature_mode == 'tracker':
                if len(pts_curr) >= self.num_min_features:
                    self.map.add_frame(curr_pose, pts_curr)
                else:
                    keypoints, descriptors = self.feature_detector.detect(frame)
                    keypoints = [keypoints[i].pt for i in range(len(keypoints))]
                    self.map.add_frame(curr_pose, keypoints, descriptors)
            else: # matcher
                self.map.add_frame(curr_pose, keypoints, descriptors)
            
            if self.feature_mode == 'matcher':
                visualize_matches(self.map, self.ground_truth_poses, self.prev_frame, frame, keypoints, matches, id, self.keyframe_step)
                self.last_matches = matches
            else:
                visualize_tracking(self.map, self.ground_truth_poses, self.prev_frame, frame, pts_prev, pts_curr, id, self.keyframe_step)
            
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

    # def run(self):
    #     for id, frame in enumerate(self.frames):
    #         self.process_frame(id, frame)
        #     if not self.process_frame(id, frame):
        #         break
        # cv2.destroyAllWindows()