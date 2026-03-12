import cv2

class FeatureMatcher():
    def __init__(self, matcher: str):
        self.matcher_name = matcher

        if self.matcher_name == 'bf':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def match(self, descriptor_1, descriptor_2):
        pass