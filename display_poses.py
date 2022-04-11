from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
camera_poses = dict()

f = open("KeyFrameTrajectory.txt", "r")

i = 0
xs = []
ys = []
zs = []
for line in f:
	parsed = line.split()
	xyz = [float(parsed[1]), float(parsed[2]), float(parsed[3])]
	quaternion = [parsed[4], parsed[5], parsed[6], parsed[7]]
	timestamp = float(parsed[0])

	r = Rotation.from_quat(quaternion)
	p = np.vstack((np.hstack((r.as_matrix(),np.array([xyz]).T)), np.array([[0.0,0.0,0.0,1]]))).astype(np.float64)

	xs.append(xyz[0])
	ys.append(xyz[2])
	zs.append(xyz[1])
	if timestamp not in camera_poses:
		camera_poses[timestamp] = p
	else:
		print("WARNING! Too many oses on the same timestamp")
	i+=1
	if i==500:
		print(p)
		break

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


X = np.array(xs)
Y = np.array(ys)
Z = np.array(zs)

scat = ax.scatter(X, Y, Z)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

plt.grid()
plt.show()