import zmq
import pickle
import zlib
import struct
import numpy as np
import ctypes
import matplotlib.pyplot as plt
from time import sleep
import imageio
import shutil
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
    ]
)
import open3d as o3d
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from image_client_movo.single_scene_graph import get_semantic_labels, label_dict
import time
from collections import defaultdict
import random
from movo.movo_utils import MOVO_PATH, MovoPolicy, MovoRobot

from pybullet_tools.utils import mesh_from_points, create_mesh, create_plane, TAN, GREY, add_line, load_pybullet

SAVE = False
FROM_SAVED = True
SEMANTIC_LABELS = True
CREATE_PB_MODEL = True


def get_color_image():
    socket.send(zlib.compress(pickle.dumps({"message_name": "get_image"})))
    message = pickle.loads(zlib.decompress(socket.recv()))
    
    if message['last']:
    	return None

    image = message['image']
    
    return image


def generate_pointcloud_default(received_images):

    # Generate the pointcloud
    pose_graph = o3d.pipelines.registration.PoseGraph()
    parameters = o3d.camera.PinholeCameraIntrinsic(960, 540, 528.612, 531.854, 477.685, 255.955)


    pcd = None
    start = True
    i = 0
    for data in received_images:
        rgb_image = np.frombuffer(data['rgb_data'], dtype=np.uint8).reshape(data['rgb_height'], data['rgb_width'], -1)
        rgb_image = rgb_image[:,:, [2,1,0]]
        rgb_image = o3d.geometry.Image(np.squeeze(np.array([rgb_image]), axis=0))
        depth_image = np.frombuffer(data['depth_data'], dtype=np.uint16).reshape(data['depth_height'], data['depth_width'], -1)
        depth_image = o3d.geometry.Image(np.array(depth_image))

        xyz = list(np.array(data["frame_pos"]))
        quaternion = data["frame_quat"]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_trunc=1000, convert_rgb_to_intensity=False)
        r = Rotation.from_quat(quaternion)
        odom = np.vstack((np.hstack((r.as_matrix(),np.array([xyz]).T)), np.array([[0.0,0.0,0.0,1.0]]))).astype(np.float64)

        if start:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                parameters)
            start = False
            i+=1
            continue

        else:
            pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                parameters)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odom))
            pcd_new.transform(pose_graph.nodes[i].pose)
            pcd += pcd_new
        i+=1
            
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    o3d.visualization.draw_geometries([pcd])


def generate_pointcloud_saved():

    pose_graph = o3d.pipelines.registration.PoseGraph()
    parameters = o3d.camera.PinholeCameraIntrinsic(960, 540, 528.612, 531.854, 477.685, 255.955)
    f1 = open("KeyFrameTrajectory.txt", "r")
    f2 = open("slam_images.txt", "r")


    start =True
    pcd = None
    i = 0
    for line in f1:
        
        line_img = f2.readline().split()

        parsed = line.split()
        xyz = [float(parsed[1]), float(parsed[2]), float(parsed[3])]
        quaternion = [parsed[4], parsed[5], parsed[6], parsed[7]]

        color = o3d.io.read_image(line_img[1])
        depth = o3d.io.read_image(line_img[3])

        if(SEMANTIC_LABELS):
        	new_colors = get_semantic_labels(color) #TODO Batch for efficiency
        	new_colors = o3d.geometry.Image((new_colors).astype(np.uint8))
        else:
        	new_colors = color
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(new_colors, depth, depth_trunc=1000, convert_rgb_to_intensity=False)

        r = Rotation.from_quat(quaternion)
        odom = np.vstack((np.hstack((r.as_matrix(),np.array([xyz]).T)), np.array([[0.0,0.0,0.0,1]]))).astype(np.float64)

        if start:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                parameters)
            start = False
            i+=1
            continue

        else:
            pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                parameters)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odom))
            pcd_new.transform(pose_graph.nodes[i].pose)
            pcd += pcd_new

        i+=1
    
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd = pcd.voxel_down_sample(voxel_size=0.001)

    if(CREATE_PB_MODEL):
        import pybullet as p
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        create_plane(mass=0, color=TAN)
        print(MOVO_PATH)
        robot_body = load_pybullet(MOVO_PATH, fixed_base=True)
        robot = MovoRobot(robot_body)
        conf = robot.get_default_conf()
        for group, positions in conf.items():
            robot.set_group_positions(group, positions)


        print("Creating pybullet model")
        print(pcd)
        pts, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
        
        # Flip and offset
        pts = pts[:, [0, 2, 1] ]
        pts[:, 2]= pts[:, 2]+0.9

        new_pts, new_colors = [], []
        wall_pts, wall_colors = [], []

        color_point_map = defaultdict(list)
        for pt, color in zip(pts, colors):
            if(pt[2]>0 and pt[2]<0.9 and random.random()<0.05):
                new_pts.append(pt)
                new_colors.append(color)
                color_point_map[tuple(color)].append(pt)
            if(pt[2]>1.2 and pt[2]<1.7 and random.random()<0.2):
                wall_pts.append(pt)
                wall_colors.append(color)

        from sklearn.cluster import DBSCAN
        color_clusters = defaultdict(list)
        colors_map = {}
        for cidx, (c, cpts) in  enumerate(color_point_map.items()):

            if(cidx==2 or cidx==3):
                colors_map[cidx]=c
                cpts = np.array(cpts)
                clustering = DBSCAN(eps=0.2, min_samples=150).fit(cpts)
                for label in np.unique(clustering.labels_):
                    if(label>=0):
                        label_idxs, = np.where(clustering.labels_ == label)
                        color_clusters[cidx].append(cpts[label_idxs, :])

        # Create meshes from these pointclouds and load them into pybullet
        print(color_clusters)
        for cluster_color, clusters in color_clusters.items():
            
            for cluster in clusters:
                obj_mesh = mesh_from_points(cluster)
                pb_body = create_mesh(
                    obj_mesh, under=True, color=list(colors_map[cluster_color])+[1]
                )

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.array(new_pts))
        new_pcd.colors = o3d.utility.Vector3dVector(np.array(new_colors))

        # Create walls as debug lines
        # wall_pts = np.array(wall_pts)
        # wall_colors = np.array(wall_colors)

        resolution = 20 
        wall_map = defaultdict(list)
        for wall_pt, wall_color in zip(wall_pts, wall_colors):
            wall_map[(int(wall_pt[0]*resolution), int(wall_pt[1]*resolution))].append(wall_color)
            
        for k, v in wall_map.items():
            if(len(v)>20):
                add_line([k[0]/float(resolution), k[1]/float(resolution), 0], [k[0]/float(resolution), k[1]/float(resolution), 3], color=GREY)

        print(wall_map)
        o3d.visualization.draw_geometries([new_pcd])

        while(True):
            import time
            time.sleep(0.1)
            p.setGravity(0,0,0)

    else:
        o3d.visualization.draw_geometries([pcd])



if __name__ == '__main__':

	# Read the arguments to the program
	if len(sys.argv) == 3:
		# If we need to save the received images to a file
		SAVE = True if sys.argv[1] == "True" else False
		# If we just compute the pointcloud from saved images
		FROM_SAVED = True if sys.argv[2] == "True" else False

	if not FROM_SAVED:
	    # Creates new communication with the server
	    context = zmq.Context()
	    socket = context.socket(zmq.REQ)
	    socket.connect("tcp://192.168.0.246:5555")

	    received_images = []
	    start = time.time()

	    print("Requesting Images")
	    while True:
	    	message = get_color_image()
	    	if message == None:
	    		break
	    	received_images.append(message)


	    end = time.time()
	    elapsed = end-start
	    print(elapsed)

	    print("Processing Images")

	    shutil.rmtree("rgb")
	    shutil.rmtree("depth")
	    os.mkdir("rgb")
	    os.mkdir("depth")

	    if os.path.exists("slam_images.txt"):
	        os.remove("slam_images.txt")

	    if os.path.exists("KeyFrameTrajectory.txt"):
	        os.remove("KeyFrameTrajectory.txt")

	    

	        # If saving is required
	    if SAVE:
	        print("Saving Images")
	        shutil.rmtree("rgb")
	        shutil.rmtree("depth")
	        os.mkdir("rgb")
	        os.mkdir("depth")

	        if os.path.exists("slam_images.txt"):
	            os.remove("slam_images.txt")



	        for data in received_images:
	            rgb_image = np.frombuffer(data['rgb_data'], dtype=np.uint8).reshape(data['rgb_height'], data['rgb_width'], -1)
	            rgb_image = rgb_image[:,:, [2,1,0]]

	            depth_image = np.frombuffer(data['depth_data'], dtype=np.uint16).reshape(data['depth_height'], data['depth_width'], -1)
	            rgb_path = 'rgb/{}.png'.format(data['rgb_stamp'])
	            depth_path = 'depth/{}.png'.format(data['depth_stamp'])
	            imageio.imwrite(rgb_path, rgb_image) 
	            imageio.imwrite(depth_path, depth_image)

	            with open('slam_images.txt', 'a') as f:
	                f.write('{} {} {} {}\n'.format(data['rgb_stamp'], rgb_path, data['depth_stamp'], depth_path))
	            with open('KeyFrameTrajectory.txt', 'a') as f:
	                f.write('{} {} {} {} {} {} {} {}\n'.format(data['frame_stamp'], data['frame_pos'][0],
	                    data['frame_pos'][1], data['frame_pos'][2],
	                    data['frame_quat'][0], data['frame_quat'][1], data['frame_quat'][2], data['frame_quat'][3]))

	    print("Generating Point Cloud")
	    generate_pointcloud_default(received_images)
	else:
		print("Generating Point Cloud")
		generate_pointcloud_saved()
