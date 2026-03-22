import cv2

class FeatureDetector():
    def __init__(self, detector='orb'):
        self.detector_name = detector
        if self.detector_name == 'orb':
            self.detector = cv2.ORB_create()
        elif self.detector_name == 'fast':
            self.detector = cv2.FastFeatureDetector_create()
        
    def detect(self, frame):
        keypoints, descriptors = None, None
        if self.detector_name == 'orb':
            keypoints, descriptors = self.detector.detectAndCompute(frame, None)
        elif self.detector_name == 'fast':
            keypoints = self.detector.detect(frame)
            descriptors = None
            # keypoints, descriptors = self.detector.compute(frame, keypoints)
        return keypoints, descriptors
        