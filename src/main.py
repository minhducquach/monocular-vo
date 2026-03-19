import cv2
from modules import *
from utils import load_kitti
import sys

from visual_odometry import VisualOdometry

DATASET_DIR = "/media/minhducquach/MiduT73/PROJECTS/monocular-vo/datasets/KITTI"
SEQUENCE = '00' # kitti sequence

if __name__ == "__main__":
    dataset = load_kitti(DATASET_DIR, SEQUENCE)
    
    frames, K, P, ground_truth = dataset['Images'], dataset['K'], dataset['P'], dataset['Homogeneous_Pose_Mat']
    
    camera = Camera(K, P)

    feature_detector = FeatureDetector(detector='orb')
    
    feature_matcher = FeatureMatcher(matcher='bf')

    visual_odometry = VisualOdometry(frames=frames, camera=camera, detector=feature_detector, matcher=feature_matcher, ground_truth=ground_truth)
    
    visual_odometry.run()

