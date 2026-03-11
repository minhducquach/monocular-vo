import cv2

class FeatureDetector():
    def __init__(self, detector='orb'):
        self.detector_name = detector
        if self.detector_name == 'orb':
            self.detector = cv2.ORB_create()
        
    def detect(self, frame):
        keypoints, descriptors = None, None
        if self.detector_name == 'orb':
            keypoints, descriptors = self.detector.detectAndCompute(frame)
        return keypoints, descriptors
        