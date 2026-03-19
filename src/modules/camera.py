import cv2
import numpy as np
from numpy.linalg import inv


class Camera():
    def __init__(self, K, P):
        self.K = K
        self.fx = K[0][0]
        self.fy = K[1][1]
        self.ux = K[0][2]
        self.uy = K[1][2]
        
        self.P = P
        self.Rt = np.dot(inv(self.K), self.P)