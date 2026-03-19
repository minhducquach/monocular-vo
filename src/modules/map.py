import cv2
import matplotlib.pyplot as plt
import numpy as np

class Map():
    def __init__(self):
        self.keyframes = []
        self.trajectory = []

    def add_frame(self, pose, keypoints, descriptors):
        keyframe = {
            'pose': pose,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        self.keyframes.append(keyframe)
        self.trajectory.append(pose)
        
    def last_frame_keypoints(self):
        return self.keyframes[-1]['keypoints']
        
    def last_frame_descriptors(self):
        return self.keyframes[-1]['descriptors']
    
    def last_frame_pose(self):
        return self.keyframes[-1]['pose']

    def get_frame(self, index):
        return self.keyframes[index]