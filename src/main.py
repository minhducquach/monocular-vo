import cv2
from modules import *
from utils import load_kitti
import sys

from visual_odometry import VisualOdometry

import configparser

config = configparser.ConfigParser()
config.read("../settings.ini")
app_config = config["app"]

DATASET_DIR = app_config["path"]
SEQUENCE = app_config["sequence"] # kitti sequence

if __name__ == "__main__":
    dataset = load_kitti(DATASET_DIR, SEQUENCE)
    
    frames, K, P, ground_truth = dataset['Images'], dataset['K'], dataset['P'], dataset['Homogeneous_Pose_Mat']
    
    camera = Camera(K, P)

    feature_detector = FeatureDetector(detector='fast')
    
    feature_matcher = FeatureMatcher(matcher='bf')
    
    feature_tracker = FeatureTracker()

    visual_odometry = VisualOdometry(camera=camera, detector=feature_detector, matcher=feature_matcher, tracker=feature_tracker, ground_truth=ground_truth, mode='tracker')
    
    for id, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        visual_odometry.process_frame(id, frame)

