import cv2

class Map():
    def __init__(self):
        self.keyframes = []

    def add_frame(self, pose, descriptor):
        keyframe = {
            'pose': pose,
            'descriptor': descriptor
        }
        self.frames.append(keyframe)