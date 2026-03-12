import cv2
from modules import *
from utils import load_kitti

from visual_odometry import VisualOdometry

DATASET_DIR = "../datasets/KITTI"
SEQUENCE = '00' # kitti sequence

if __name__ == "__main__":
    dataset = load_kitti(DATASET_DIR, SEQUENCE)
    
    frames, K = dataset['Images'], dataset['K']
    
    camera = Camera(K)

    feature_detector = FeatureDetector(detector='orb')
    
    feature_matcher = FeatureMatcher(matcher='bf')

    visual_odometry = VisualOdometry(frames=frames, camera=camera, detector=feature_detector, matcher=feature_matcher)
    
    visual_odometry.run()

