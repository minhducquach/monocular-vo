import cv2
import numpy as np

def init_pose():
    pose = np.eye(4)
    pose[:, 3] = 1
    return pose

def choose_next_frame(id, keyframe_step):
    if (id + 1) % keyframe_step == 0:
        return True
    return False

if __name__ == "__main__":
    print(init_pose())