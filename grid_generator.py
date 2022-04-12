import open3d as o3d
import numpy as np
import os
import sys

import matplotlib.pyplot as plt


def convert_point_to_pixel(point, grid_shape, min_bounds, dimensions):

	u,v = point[0], point[2]
	u = int(((u - min_bounds[0])/dimensions[0])*(grid_shape[0]-1))
	v = int(((v - min_bounds[1])/dimensions[1])*(grid_shape[1]-1))

	return u,v


ratio = 10

pcd = o3d.io.read_point_cloud("generated_pointcloud.pcd")



min_bounds = pcd.get_min_bound()
min_bounds = [min_bounds[0], min_bounds[2]]

max_bounds = pcd.get_max_bound()
max_bounds = [max_bounds[0], max_bounds[2]]


dimensions_world = ((abs(min_bounds[0])) + (abs(max_bounds[0])),
			        (abs(min_bounds[1])) + (abs(max_bounds[1])))


dimensions_pixel = ((int(abs(min_bounds[0])) + int(abs(max_bounds[0])))*10,
			        (int(abs(min_bounds[1])) + int(abs(max_bounds[1])))*10)



# Just leave space that the robot can collide with
bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bounds[0], -1.02, min_bounds[1])
	, max_bound=(max_bounds[0], 0.1, max_bounds[1]))
pcd = pcd.crop(bb)


cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30,
                                                    std_ratio=5.0)
pcd = pcd.select_by_index(ind)

#o3d.visualization.draw_geometries([pcd])



# Get the points and compute occupancy grid
# TRY TO MAKE IT MOPRE EFFICIENT
points = np.asarray(pcd.points)

grid_map = np.zeros(dimensions_pixel)


for point in points:
	u,v = convert_point_to_pixel(point, grid_map.shape, min_bounds, dimensions_world)

	grid_map[u,v]+=1

grid_map[grid_map <= 20] = 0
grid_map[grid_map > 20] = 1


plt.imshow(grid_map)
plt.show()
