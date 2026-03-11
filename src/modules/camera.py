import cv2

class Camera():
    def __init__(self, K):
        self.K = K
        self.fx = K[0][0]
        self.fy = K[1][1]
        self.ux = K[0][2]
        self.uy = K[1][2]