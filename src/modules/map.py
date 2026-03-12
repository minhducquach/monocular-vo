import cv2

class Map():
    def __init__(self):
        self.keyframes = []

    def add_frame(self, pose, keypoints, descriptors):
        keyframe = {
            'pose': pose,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        self.keyframes.append(keyframe)
        
    def last_frame_keypoints(self):
        return self.keyframes[-1]['keypoints']
        
    def last_frame_descriptors(self):
        return self.keyframes[-1]['descriptors']
    
    def last_frame_pose(self):
        return self.keyframes[-1]['pose']