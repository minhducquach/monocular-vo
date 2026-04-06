import cv2
import matplotlib.pyplot as plt
import numpy as np

class Map():
    def __init__(self):
        self.keyframes = []
        self.trajectory = []
        self.landmarks = []

    def add_frame(self, pose, keypoints, descriptors=None):
        keyframe = {
            'pose': pose,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        
        self.keyframes.append(keyframe)
        self.trajectory.append(pose)
        
    def add_landmarks(self, landmarks):
        for landmark in landmarks:
            self.landmarks.append(landmark)
        
    def last_frame_keypoints(self):
        return self.keyframes[-1]['keypoints']
        
    def last_frame_descriptors(self):
        return self.keyframes[-1]['descriptors']
    
    def last_frame_pose(self):
        return self.keyframes[-1]['pose']

    def get_frame(self, index):
        return self.keyframes[index]