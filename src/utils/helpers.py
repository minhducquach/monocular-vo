import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive mode
fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(16, 6))

def init_pose():
    pose = np.eye(4)
    return pose

def choose_next_frame(id, keyframe_step):
    if (id + 1) % keyframe_step == 0:
        return True
    return False

def compute_relative_scale(camera, map, prev_matches, curr_matches, curr_keypoints, curr_T):
    keypoints_0 = map.get_frame(-2)['keypoints'] # Frame t-2
    keypoints_1 = map.get_frame(-1)['keypoints'] # Frame t-1
    keypoints_2 = curr_keypoints

    pose_0 = map.get_frame(-2)['pose']
    pose_1 = map.get_frame(-1)['pose']

    f1_to_f0 = {m.trainIdx: m.queryIdx for m in prev_matches}

    tracked_pts_0 = []
    tracked_pts_1 = []
    tracked_pts_2 = []

    for m2 in curr_matches:
        idx_f1 = m2.queryIdx
        idx_f2 = m2.trainIdx
        
        if idx_f1 in f1_to_f0:
            idx_f0 = f1_to_f0[idx_f1]
            
            tracked_pts_0.append(keypoints_0[idx_f0].pt)
            tracked_pts_1.append(keypoints_1[idx_f1].pt)
            tracked_pts_2.append(keypoints_2[idx_f2].pt)

    if len(tracked_pts_0) < 2:
        return 1

    pts0 = np.array(tracked_pts_0, dtype=np.float32)
    pts1 = np.array(tracked_pts_1, dtype=np.float32)
    pts2 = np.array(tracked_pts_2, dtype=np.float32)

    P0 = camera.K @ np.linalg.inv(pose_0)[:3, :4]
    P1 = camera.K @ np.linalg.inv(pose_1)[:3, :4]
    P2 = camera.K @ np.linalg.inv(pose_1 @ curr_T)[:3, :4]

    scales = []

    for i in range(len(pts0)-1):
        p1_1 = cv2.triangulatePoints(P0, P1, pts0[i], pts1[i])
        p1_1 = p1_1 / p1_1[3]
        p1_2 = cv2.triangulatePoints(P1, P2, pts1[i], pts2[i])
        p1_2 = p1_2 / p1_2[3]

        p2_1 = cv2.triangulatePoints(P0, P1, pts0[i+1], pts1[i+1])
        p2_1 = p2_1 / p2_1[3]
        p2_2 = cv2.triangulatePoints(P1, P2, pts1[i+1], pts2[i+1])
        p2_2 = p2_2 / p2_2[3]

        dist_old = np.linalg.norm(p2_1.T - p1_1.T)
        dist_new = np.linalg.norm(p2_2.T - p1_2.T)

        scales.append(dist_old / dist_new)
    
    return np.median(scales)

def compute_absolute_scale(gt, id, step):
    prev = gt[id - step, :3, 3]
    curr = gt[id, :3, 3]
    
    return np.linalg.norm(prev - curr)
    

def visualize(map, ground_truth, prev_frame, frame, keypoints, matches, id, step, mode):
    # match_img = cv2.drawMatches(self.prev_frame, self.map.last_frame_keypoints(), frame, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
    # cv2.imshow('img', match_img)
    plt.cla()

    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Z")
    ax_traj.set_title("Monocular Visual Odometry Trajectory (X-Z)")
    ax_traj.axis("equal")
    ax_traj.grid(True)
    ax_traj.legend()
    
    if mode == 'matcher':
        img_match = cv2.drawMatches(prev_frame, map.last_frame_keypoints(), frame, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        ax_img.clear()
        ax_img.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"Feature Matches (Frame {id})")
        ax_img.axis("off")
    else:
        pass
    
    traj_np = np.array(map.trajectory)
    max_true_frame_id = len(traj_np) * step

    gt_x = ground_truth[:max_true_frame_id:step, 0, 3]
    gt_z = ground_truth[:max_true_frame_id:step, 2, 3]

    line_gt, = ax_traj.plot(gt_x, gt_z, 'g-', label='Ground Truth')
    line_traj, = ax_traj.plot([], [], 'b-', label='Trajectory')
    
    line_traj.set_data(traj_np[:, 0, 3], traj_np[:, 2, 3])
    line_gt.set_data(gt_x, gt_z)

    ax_traj.relim()
    ax_traj.autoscale_view()
    ax_traj.legend()
    
    plt.pause(0.01)
    
if __name__ == "__main__":
    print(init_pose())