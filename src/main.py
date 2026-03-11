import cv2
from modules import *
from utils import load_kitti

from visual_odometry import VisualOdometry

DATASET_DIR = "/media/minhducquach/MiduT73/PROJECTS/monocular-vo/datasets/KITTI"
SEQUENCE = '00' # kitti sequence

if __name__ == "__main__":
    dataset = load_kitti(DATASET_DIR, SEQUENCE)
    
    frames, K = dataset['Images'], dataset['K']
    
    camera = Camera(K)

    visual_odometry = VisualOdometry(frames=frames, camera=camera, detector=None, matcher=None, map=None)
    
    visual_odometry.run()

