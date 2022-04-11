from util.camera_pose_visualizer import CameraPoseVisualizer
import numpy as np
from scipy.spatial.transform import Rotation

camera_poses = dict()

f = open("../CameraTrajectory.txt", "r")

for line in f:
	parsed = line.split()
	xyz = [parsed[1], parsed[2], parsed[3]]
	quaternion = [parsed[4], parsed[5], parsed[6], parsed[7]]
	timestamp = float(parsed[0])
	
	if timestamp not in camera_poses:
		camera_poses[timestamp] = [xyz, quaternion]
	else:
		print("WARNING! Too poses on the same timestamp")

	r = Rotation.from_quat(quaternion)
	p = np.vstack((np.hstack((r.as_matrix(),np.array([xyz]).T)), np.array([[0.0,0.0,0.0,1.0]]))).astype(np.float64)

print(p)


visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [-100, 100])
visualizer.extrinsic2pyramid(p, 'c', 10)
visualizer.show()

