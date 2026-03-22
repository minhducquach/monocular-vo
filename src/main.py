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
    
    feature_tracker = FeatureTracker()

    visual_odometry = VisualOdometry(camera=camera, detector=feature_detector, matcher=feature_matcher, tracker=feature_tracker, ground_truth=ground_truth, mode='tracker')
    
    for id, frame in enumerate(frames):
        visual_odometry.process_frame(id, frame)

