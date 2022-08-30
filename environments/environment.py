from pybullet_planning.pybullet_tools.utils import (LockRenderer, load_pybullet, set_joint_positions, joint_from_name,
                                                    Point, Pose, Euler,
                                                    set_pose, create_box, TAN, get_link_pose,
                                                    get_camera_matrix, get_image_at_pose, tform_point, invert, multiply,
                                                    pixel_from_point, AABB, OOBB, BLUE, RED, YELLOW, link_from_name,
                                                    aabb_contains_point, aabb_from_oobb,
                                                    get_aabb, RGBA, recenter_oobb, get_aabb, draw_oobb,
                                                    aabb_from_points, OOBB,
                                                    aabb_union, aabb_overlap, scale_aabb, get_aabb_center, 
                                                    draw_aabb, aabb_intersection,
                                                    get_aabb_volume)
from pybullet_planning.pybullet_tools.voxels import (VoxelGrid)
from utils.motion_planning_interface import DEFAULT_JOINTS
from utils.utils import iterate_point_cloud
import pybullet as p
import os
import numpy as np
from collections import namedtuple, defaultdict
import functools
from scipy.spatial.transform import Rotation as R
from abc import ABC

GRID_HEIGHT = 2  # Height of the visibility and occupancy grids
GRID_RESOLUTION = 0.2  # Grid resolutions

LIGHT_GREY = RGBA(0.7, 0.7, 0.7, 1)

Room = namedtuple("Room", ["walls", "floors", "aabb", "movable_obstacles"])
Force = namedtuple("Force", ["magnitude", "angle"])

fx = 80
fy = 80
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 128
CAMERA_MATRIX = get_camera_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, fx, fy)
FAR = 3

class Environment(ABC):

    def __init__(self):
        pass

    def update_movable_boxes(self, camera_image, ignore_obstacles=[], **kwargs):
        """
        Given an image from the camera updates all the detected objects that are labeled as movable.

        Args:
            camera_image: The taken image from the camera.
            ignore_obstacles (list): A list of obstacles that will not be taken into account when updating.
        Returns:
            set: A set of voxels from the vision grid if any was updated.
        """
        relevant_cloud = [lp for lp in iterate_point_cloud(camera_image, **kwargs)
                          if aabb_contains_point(lp.point, self.room.aabb)
                          ]

        object_points = defaultdict(list)
        for labeled_point in relevant_cloud:
            if (labeled_point.label[0] in self.room.movable_obstacles and
                    labeled_point.label[0] not in ignore_obstacles):
                object_points[labeled_point.label[0]].append(labeled_point.point)

        # Convert point clusters into bounding boxes
        new_boxes = []
        for _, points in object_points.items():
            new_boxes.append(OOBB(aabb_from_points(points), Pose()))

        # If a bounding box intersects with an existing one, replace with union
        def test_in(box, boxes):
            return any([(all(np.array(box.aabb.upper) == np.array(q.aabb.upper)) and
                         all(np.array(box.aabb.lower) == np.array(q.aabb.lower))) for q in boxes])

        all_new_boxes = []
        overlapped_boxes = []
        for movable_box in self.movable_boxes:
            for new_box in new_boxes:
                if aabb_overlap(movable_box.aabb, new_box.aabb):
                    all_new_boxes.append(OOBB(aabb_union([movable_box.aabb, new_box.aabb]), Pose()))
                    overlapped_boxes.append(movable_box)
                    overlapped_boxes.append(new_box)
        for b in new_boxes + self.movable_boxes:
            if not (test_in(b, overlapped_boxes)):
                all_new_boxes.append(b)

        # Merge all if there is repetition. Should not happen often.
        merging = True
        length = len(all_new_boxes)
        indexes = set(range(length))
        while merging:
            merging = False
            for box1_i in indexes:
                box1 = all_new_boxes[box1_i]
                for box2_i in indexes:
                    box2 = all_new_boxes[box2_i]
                    if box1_i == box2_i:
                        continue
                    if aabb_overlap(box1.aabb, box2.aabb):
                        all_new_boxes.append(OOBB(aabb_union([box1.aabb, box2.aabb]), Pose()))
                        indexes.add(length)
                        length += 1
                        merging = True
                        indexes.remove(box1_i)
                        indexes.remove(box2_i)
                        break
                if merging:
                    break
        all_new_boxes = [all_new_boxes[i] for i in indexes]

        # Remove points from occupancy/visibility grids
        vision_update = set()
        for movable_box in all_new_boxes:
            for voxel in self.occupancy_grid.voxels_from_aabb(scale_aabb(movable_box.aabb, 1.3)):
                self.occupancy_grid.set_free(voxel)
                if self.static_vis_grid.contains(voxel):
                    self.visibility_grid.set_free(voxel)
                    vision_update.add(voxel)

        self.movable_boxes = all_new_boxes

        return vision_update


    def update_occupancy(self, q, camera_image, ignore_obstacles=[], **kwargs):
        """
        Updates the occupancy grid based on an image input.

        Args:
            camera_image: The taken image from the camera.
            ignore_obstacles (list): A list of obstacles that will not be taken into account when updating.
        """
        occ_aabb = AABB(lower=(self.room.aabb[0][0]-1, self.room.aabb[0][1]-1, self.room.aabb[0][2]),
                        upper=(self.room.aabb[1][0]+1, self.room.aabb[1][1]+1, self.room.aabb[1][2]))
        relevant_cloud = [lp for lp in iterate_point_cloud(camera_image, **kwargs)
                          if aabb_contains_point(lp.point, occ_aabb)]

        # Get collision from vision
        for labeled_point in relevant_cloud:
            if labeled_point.label[0] not in ignore_obstacles:
                if labeled_point.label[1] == -1:
                    point = labeled_point.point
                    self.occupancy_grid.add_point(point)
        # Get collision from lidar
        for voxel in self.lidar_scan(q):
            self.occupancy_grid.set_occupied(voxel)

    def disconnect(self):
        try:
            p.disconnect()
        except:
            pass
        
    def lidar_scan(self, q, radius=5, angle=2*np.pi):
        """
        Gets the obstruction voxels from a lidar scan at a certain distance and
        angle from the base of the robot.

        Args:
            q (tuple): Center of the scan.
            radius (float): Radius of the scan.
            angle (float): Swept angle. The bisector of the angle will always be pointing
                           towards the front of the robot. Between 0 and 2PI.
        Returns:
            set: A set of voxels representing the obstructions found from the lidar scan.
        """
        # Get the angles to step to find collisions
        step = np.pi/36
        angles = [q[2] + step*x for x in range(int(angle/(2*step))+1)]
        angles += [q[2] - step*x for x in range(1, int(angle/(2*step))+1)]

        voxels = set()
        for angle in angles:
            # Get point along the circumference
            vox_q = q + np.array([radius*np.cos(angle), radius*np.sin(angle), 0])
            vox_q[2] = 0.25
            vox = self.occupancy_grid.voxel_from_point(vox_q)
            robot_vox = self.occupancy_grid.voxel_from_point([q[0], q[1], 0.25])
            # Ray cast to the intended position in the circumference.
            path = self.check_voxel_path_coll(robot_vox, vox, self.occupancy_grid)
            if not path[1]:
                flag = False
                for voxel in path[0]:
                    if flag:
                        break
                    # Check for the first obstacle to intersect.
                    vox_aabb = self.occupancy_grid.aabb_from_voxel(voxel)
                    vox_aabb = AABB(lower=vox_aabb[0] + np.ones(3) * 0.01,
                                    upper=vox_aabb[1] - np.ones(3) * 0.01)
                    for elem in self.room.walls + self.objects:
                        obj_aabb = get_aabb(elem)
                        if aabb_overlap(obj_aabb, vox_aabb):
                            voxels.add(voxel)
                            flag = True
                            break
        return voxels

    def update_visibility(self, camera_pose, camera_image, q):
        """
        Updates the visibility grid based on a camera image.

        Args:
            camera_pose (tuple): Pose of the camera in the world frame.
            camera_image: The taken image from the camera.
            q (tuple): Robot configuration corresponding to the taken image.
        Returns:
            set: The gained vision obtaining from the given image.
        """
        surface_aabb = self.visibility_grid.aabb
        camera_pose, camera_matrix = camera_image[-2:]
        grid = self.visibility_grid
        self.gained_vision[q] = set()

        # For each voxel in the grid, check whether it was seen in the image
        for voxel in grid.voxels_from_aabb(surface_aabb):
            center_world = grid.to_world(grid.center_from_voxel(voxel))
            center_camera = tform_point(invert(camera_pose), center_world)
            distance = center_camera[2]
            pixel = pixel_from_point(CAMERA_MATRIX, center_camera)
            if pixel is not None:
                r, c = pixel
                depth = camera_image.depthPixels[r, c]
                if distance <= depth:
                    self.viewed_voxels.append(voxel)
                    grid.set_free(voxel)
                    self.gained_vision[q].add(voxel)
        return self.gained_vision[q]


    def setup_grids(self):
        """
        Setups the occupancy, visibility, and movable objects grids.
        """
        self.setup_occupancy_grid()
        self.setup_visibility_grid()
        self.setup_movable_boxes()



    def setup_visibility_grid(self):
        """
        Setups the visibility grids.
        """
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = Pose(Point(z=0.01))
        surface_aabb = AABB(lower=self.room.aabb.lower+np.array([-1, -1, 0]),
                            upper=(self.room.aabb.upper[0]+1, self.room.aabb.upper[1]+1, GRID_RESOLUTION))
        # Defines two grids, one for visualization, and a second one for keeping track of regions during
        # planning.
        grid = VoxelGrid(
            resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=BLUE
        )
        static_grid = VoxelGrid(
            resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=BLUE
        )
        for voxel in grid.voxels_from_aabb(surface_aabb):
            grid.set_occupied(voxel)
            static_grid.set_occupied(voxel)

        self.visibility_grid = grid
        self.static_vis_grid = static_grid
        self.gained_vision = dict()


    def setup_default_vision(self, G):
        """
        Setups a map of base configurations to viewed voxels.

        Args:
            G (object): Graph object defining how our space is discretized.
        """
        # Since movements are currently defined as translations along the axis, we can use a set of default
        # visions and only translating in space to improve the running time.
        self.default_vision = dict()
        for i in range(G.t_step):
            q = (0, 0, round(i*2*np.pi/G.t_step, 3))
            self.default_vision[q] = self.gained_vision_from_conf(q)


    def get_optimistic_path_vision(self, path, G, attachment=None):
        """
        Gets optimistic vision along a path. Optimistic vision is defined as the vision cone from a certain
        configuration, unless the configuration has already been viewed and the actual vision is known.

        Args:
            path (list): The path to which vision is to be determined.
            G (object): Graph object defining how our space is discretized.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            set: A set of voxels that correspond to the gained vision.
        """
        vision = set()
        for q in path:
            vision.update(self.get_optimistic_vision(q, G, attachment=attachment))
        return vision

    @functools.lru_cache(typed=False)
    def get_icp(self, q):
        pose = Pose(point=Point(x=q[0], y=q[1], z=0), euler=[0, 0, q[2]])
        camera_pose = multiply(pose, self.camera_pose)
        ip = invert(camera_pose)
        icp_r = R.from_quat(ip[1]).as_matrix()
        return icp_r, np.array(ip[0]) 

    def in_view_cone(self, points, path, attachment=None):
        if len(points) == 0:
            return True

        paths_rays = []
        for q in path:
            icp_r, icp_p = self.get_icp(q)
            rays = icp_r.dot(points.T).T+icp_p
            mag = np.expand_dims(rays[:, 2], 1)
            s = CAMERA_MATRIX.dot((rays / mag).T)[:2, :]
            paths_rays.append(np.expand_dims(np.all(((s > 0) & (s < CAMERA_HEIGHT)), axis=0), axis=0))

        paths_rays = np.concatenate(paths_rays, axis=0).astype(np.uint8)
        return np.min(np.sum(paths_rays, axis=0)) > 0
                    
    def connect(self):
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)



    def get_optimistic_vision(self, q, G, attachment=None, obstructions=set()):
        """
        Gets the optimistic vision of a specified configuration. Optimistic vision is defined as
        the vision cone from a certain configuration, unless the configuration has already been
        viewed and the actual vision is known.

        Args:
            q (tuple): The robot configuration.
            G (object): Graph object defining how our space is discretized.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            set: A set of voxels that correspond to the gained vision.
        """
        # If we have already reached the configuration, return the obtained vision.
        # Only if no object is attached.
        if q in self.gained_vision and attachment is None:
            return self.gained_vision[q]

        # Look for the corresponding default vision and transform the voxels accordingly.
        default_one = self.default_vision[(0, 0, round(q[2], 3))]
        resulting_voxels = set()
        for voxel in default_one:
            voxel_w = np.array(voxel)*np.array(G.res)
            new_voxel_w = voxel_w + np.array([q[0], q[1], GRID_RESOLUTION])
            new_voxel = np.rint(np.array(new_voxel_w)/np.array(G.res))
            new_voxel = (new_voxel[0], new_voxel[1], 0)
            if self.static_vis_grid.contains(new_voxel):
                resulting_voxels.add(new_voxel)

        # If an object is attached, compute set of voxels that could occlude vision.
        extra_obs = set()
        extra_obs.update(obstructions)
        if attachment is not None:
            obj_oobb = self.movable_object_oobb_from_q(attachment[0], q, attachment[1])
            extra_obs.update(set([x for x in self.occupancy_grid.voxels_from_aabb(obj_oobb.aabb)]))

        # Get the occlusion given by the movable objects
        for obj in self.movable_boxes:
            extra_obs.update(set([x for x in self.occupancy_grid.voxels_from_aabb(obj.aabb)]))


        # Only filter those voxels that are not obstructed by an occupied voxel
        # that has already been detected
        final_voxels = set()
        pose = Pose(point=Point(x=q[0], y=q[1], z=0), euler=[0, 0, q[2]])
        camera_pose = multiply(pose, self.camera_pose)
        for voxel in resulting_voxels:
            result = self.check_voxel_path_coll(voxel,
                                                self.occupancy_grid.voxel_from_point(camera_pose[0]),
                                                self.occupancy_grid,
                                                extra_obs=extra_obs)
            if not result[1]:
                final_voxels.add(voxel)

        return final_voxels

    def check_voxel_path_coll(self, start_cell, goal_cell, grid, extra_obs=set()):
        """
        Check for path between two cells in a grid using Bresenham's Algorithm and decide
        if a cell in the path is occupied, therefore colliding.

        Args:
            start_cell (tuple): the starting voxel.
            goal_cell (tuple): the goal voxel.
            grid (object): the voxel grid where the voxels are present.
            extra_obs (set): An extra set of voxels that could also obstruct the view.
        Returns:
            - The path from start to goal voxel and a bool representing whether a collision
              was found.
        """
        x1, y1, z1 = start_cell
        x2, y2, z2 = goal_cell
        ListOfPoints = []
        if grid.contains((x1, y1, z1)) or (x1, y1,z1) in extra_obs:
            return ListOfPoints, True
        ListOfPoints.append((x1, y1, z1))
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        if (x2 > x1):
            xs = 1
        else:
            xs = -1
        if (y2 > y1):
            ys = 1
        else:
            ys = -1
        if (z2 > z1):
            zs = 1
        else:
            zs = -1

        # Driving axis is X-axis"
        if (dx >= dy and dx >= dz):
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while (x1 != x2):
                x1 += xs
                if (p1 >= 0):
                    y1 += ys
                    p1 -= 2 * dx
                if (p2 >= 0):
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                ListOfPoints.append((x1, y1, z1))
                if grid.contains((x1, y1, z1)) or (x1, y1,z1) in extra_obs:
                    return ListOfPoints, True

        # Driving axis is Y-axis"
        elif (dy >= dx and dy >= dz):
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while (y1 != y2):
                y1 += ys
                if (p1 >= 0):
                    x1 += xs
                    p1 -= 2 * dy
                if (p2 >= 0):
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                ListOfPoints.append((x1, y1, z1))
                if grid.contains((x1, y1, z1)) or (x1, y1,z1) in extra_obs:
                    return ListOfPoints, True

        # Driving axis is Z-axis"
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while (z1 != z2):
                z1 += zs
                if (p1 >= 0):
                    y1 += ys
                    p1 -= 2 * dz
                if (p2 >= 0):
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                ListOfPoints.append((x1, y1, z1))
                if grid.contains((x1, y1, z1)) or (x1, y1,z1) in extra_obs:
                    return ListOfPoints, True

        return ListOfPoints, False

    def find_path_movable_obstruction(self, path):
        """
        Find the first movable object bounding box that this path collides with.

        Args:
            path (list): The path of the robot to take.
        Returns:
            The first movable object the path collides with.
        """
        best_obj = None
        best_volume = 0
        for q in path:
            aabb = self.aabb_from_q(q)
            for movable_box in self.movable_boxes:
                intersec = aabb_intersection(aabb, movable_box.aabb)
                if intersec is not None:
                    volume = get_aabb_volume(intersec)
                    if volume > best_volume:
                        best_volume = volume
                        best_obj = movable_box
        return best_obj
    def sample_attachment_push(self, obj, G, obstructions=set()):
        """
        Helper function to sample an attachment position for the case that the object can
        only be pushed.

        Args:
            obj (int): The object index in the environment
            G: The discrete graph of the environment.
        Returns:
            A set of poses that are valid attachments
        """
        aabb = get_aabb(obj)
        mid_point = (np.array(aabb.lower) + np.array(aabb.upper)) / 2

        attachments = [[round(mid_point[0], 1), aabb.lower[1], round(np.pi / 2, 3)],
                       [round(mid_point[0], 1), aabb.upper[1], round(3 * np.pi / 2, 3)],
                       [aabb.lower[0], round(mid_point[1], 1), 0],
                       [aabb.upper[0], round(mid_point[1], 1), round(np.pi, 3)]]

        # Fit the attachments to the grid
        for i in range(len(attachments)):
            robot_aabb = self.oobb_from_q(attachments[i]).aabb
            if i == 0:
                attachments[i][1] = attachments[i][1] - (robot_aabb[1][0] + G.res[1]) + G.res[1]/2
                attachments[i][1] = round(attachments[i][1] - (attachments[i][1] % G.res[1]), 1)
            elif i == 1:
                attachments[i][1] = attachments[i][1] + (robot_aabb[1][0] + G.res[1]) + G.res[1]/2
                attachments[i][1] = round(attachments[i][1] - (attachments[i][1] % G.res[1]), 1)
            elif i == 2:
                attachments[i][0] = attachments[i][0] - (robot_aabb[1][0] + G.res[0]) + G.res[0]/2
                attachments[i][0] = round(attachments[i][0] - (attachments[i][0] % G.res[0]), 1)
            else:
                attachments[i][0] = attachments[i][0] + (robot_aabb[1][0] + G.res[0]) + G.res[0]/2
                attachments[i][0] = round(attachments[i][0] - (attachments[i][0] % G.res[0]), 1)
            if i < 2:
                attachments[i][0] = attachments[i][0] + G.res[0] / 2
                attachments[i][0] = round(attachments[i][0] - (attachments[i][0] % G.res[0]), 1)
            else:
                attachments[i][1] = attachments[i][1] + G.res[1] / 2
                attachments[i][1] = round(attachments[i][1] - (attachments[i][1] % G.res[1]), 1)

        positions = set()
        for pos in attachments:
            obst, coll = self.obstruction_from_path([tuple(pos)], obstructions)
            if len(obst) == 0 and coll is None:
                positions.add(tuple(pos))
        return positions


    def sample_attachment_poses(self, movable_object, G, radius=1, obstructions=set()):
        """
        Given an object to move, sample valid poses around it for a successful attachment

        Args:
            movable_object : The object we which to attach to. Given as an OOBB.
            G : The discrete graph of the environment.
            radius: How far from the object we are allowed to attach.
        Returns:
            A set of poses that are valid attachments
        """
        # Check whether the object is push only, and handle it accordingly.
        obj = None
        for obj in self.objects:
            if aabb_contains_point(get_aabb_center(movable_object.aabb), get_aabb(obj)):
                break

        if obj in self.push_only:
            return self.sample_attachment_push(obj, G, obstructions=obstructions)

        # If the object can be moved freely, then sample attachments
        mid_point = ((movable_object.aabb[0][0] + movable_object.aabb[1][0]) / 2,
                     (movable_object.aabb[0][1] + movable_object.aabb[1][1]) / 2)
        mid_node = (mid_point[0] + G.res[0] / 2, mid_point[1] + G.res[1] / 2)
        mid_node = (round(mid_node[0] - (mid_node[0] % G.res[0]), 2),
                    round(mid_node[1] - (mid_node[1] % G.res[1]), 2), 0)

        queue = [mid_node]
        expanded = set()
        candidates = set()
        while queue:
            current = queue.pop(0)
            if current in expanded:
                continue
            expanded.add(current)
            for q_i in G.neighbors[G.vex2idx[current]]:
                q = G.vertices[q_i]
                if distance(q, mid_point) <= radius:
                    if q in candidates:
                        continue
                    queue.append(q)
                    obst, coll = self.obstruction_from_path([q], obstructions)
                    if aabb_overlap(scale_aabb(movable_object.aabb, 1.1), self.aabb_from_q(q)) or\
                            len(obst) != 0 or coll is not None:
                        continue
                    angle = np.arctan2(mid_point[1] - q[1], mid_point[0]-q[0])
                    if abs(find_min_angle(angle, q[2])) < np.pi/18:
                        # TODO: ELIMINATE THIS FOR GENERAL OBJECTS.
                        if q[0] >= mid_point[0]-0.5:
                            continue
                        candidates.add(q)
        return candidates

    def sample_placement(self, q_start, coll_obj, G, p_through, obstructions=set()):
        """
        Samples a placement position for an object such that it does not collide with a given path

        Args:
            q_start (tuple): Configuration of the robot at the start of attachment.
            coll_obj : The oobb of the attached object.
            G : THe discrete representation of the configuration space.
            p_through (set): The set of voxels of the path we can't collide with.
        Returns:
            A random configuration that is valid, the grasp transform, as well as the corresponding object.
        """
        # Look for which object the oobb corresponds to in the environment.
        obj = None
        for obj in self.objects:
            if aabb_contains_point(get_aabb_center(coll_obj.aabb), get_aabb(obj)):
                break
        # If the object can only be pushed then restrict its possible placements
        if obj in self.push_only:
            return self.sample_push_placement(q_start, coll_obj, G, p_through, obj,
                                              obstructions=obstructions)

        # Compute the grasp transform of the attachment.
        base_pose = Pose(
            point=Point(x=q_start[0], y=q_start[1]),
            euler=Euler(yaw=q_start[2]),
        )
        obj_pose = Pose(point=get_aabb_center(coll_obj.aabb))
        base_grasp = multiply(invert(base_pose), obj_pose)
        good = False
        # Look for a random configuration and return it only if it is valid.
        while not good:
            rand_q = G.rand_vex(self)
            if aabb_overlap(coll_obj.aabb, self.aabb_from_q(rand_q)):
                continue
            obst, coll = self.obstruction_from_path([rand_q], p_through.union(obstructions),
                                                    attachment=[coll_obj, base_grasp])
            if len(obst) == 0 and coll is None:
                return rand_q, base_grasp, obj

    def sample_push_placement(self, q_start, coll_obj, G, p_through, obj, obstructions=set()):
        """
        Helper function for finding a placement position for an object that can only be pushed

        Args:
            q_start (tuple): Configuration of the robot at the start of attachment.
            coll_obj : The oobb of the attached object.
            G : THe discrete representation of the configuration space.
            p_through (set): The path we can't collide with.
            obj (int): The representation of the object in the environment.
        Returns:
            A random configuration that is valid, the grasp transform, as well as the corresponding object.
        """
        room_aabb = self.room.aabb
        base_pose = Pose(
            point=Point(x=q_start[0], y=q_start[1]),
            euler=Euler(yaw=q_start[2]),
        )
        obj_pose = Pose(point=get_aabb_center(coll_obj.aabb))
        base_grasp = multiply(invert(base_pose), obj_pose)
        placements = []
        if q_start[2] == 0:
            placements = [(round(x, 1), q_start[1], q_start[2])
                          for x in np.arange(q_start[0], room_aabb[1][0], G.res[0])]
        elif q_start[2] == round(np.pi/2, 3):
            placements = [(q_start[0], round(x, 1), q_start[2])
                          for x in np.arange(q_start[1], room_aabb[1][1], G.res[1])]
        elif q_start[2] == round(np.pi, 3):
            placements = [(round(x, 1), q_start[1], q_start[2])
                          for x in np.arange(q_start[0], room_aabb[0][0], -G.res[0])]
        elif q_start[2] == round(3*np.pi/2, 3):
            placements = [(q_start[0], round(x, 1), q_start[2])
                          for x in np.arange(q_start[1], room_aabb[0][1], -G.res[1])]

        good = False
        idxs = np.random.randint(0, len(placements), size=len(placements))
        for idx in idxs:
            rand_q = placements[idx]
            if aabb_overlap(coll_obj.aabb, self.aabb_from_q(rand_q)):
                continue
            obst, coll = self.obstruction_from_path([rand_q], p_through.union(obstructions),
                                                    attachment=[coll_obj, base_grasp])
            if len(obst) == 0 and coll is None:
                return rand_q, base_grasp, obj
        return None, None, None


    def movable_object_oobb_from_q(self, movable_object_oobb, q, grasp, visualize=False):
        """
        Compute the oobb of an attached object given a robot configuration.

        Args:
            movable_object_oobb : The original oobb of the object when attachment began.
            q (tuple): The desired configuration of the robot.
            grasp : The trasnform from robot configuration to object's pose.
            visualize (bool): Whether to visualize the resulting aabbs after the transform.
        Returns:
            The new oobb of the object after the transform.
        """
        robot_aabb = self.aabb_from_q(q)
        robot_pose = Pose(
            point=Point(x=q[0], y=q[1]),
            euler=Euler(yaw=q[2]),
        )
        obj_pose = multiply(robot_pose, grasp)
        aabb_object, _ = recenter_oobb((movable_object_oobb.aabb, obj_pose))
        object_aabb = AABB(
            lower=[aabb_object[0][0] + obj_pose[0][0], aabb_object[0][1] + obj_pose[0][1],
                   aabb_object[0][2] + obj_pose[0][2]],
            upper=[aabb_object[1][0] + obj_pose[0][0], aabb_object[1][1] + obj_pose[0][1],
                   aabb_object[1][2] + obj_pose[0][2]])
        if visualize:
            draw_aabb(robot_aabb)
            draw_aabb(object_aabb)
        new_object = OOBB(aabb=object_aabb, pose=Pose())
        return new_object


    def clear_noise_from_attached(self, q, attachment):
        """
        Clears any obstructions and visualization noise caused by moving with an attached object

        Args:
            q (tuple): The robot configuration that has to be cleaned.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        """
        # Remove the image from cache, since it is not the real free vision.
        self.gained_vision.pop(q)

        obj_oobb = self.movable_object_oobb_from_q(attachment[0], q, attachment[1])

        # Eliminate any movable object that could have originated from the attached object
        new_l = []
        for oobb in self.movable_boxes:
            if not aabb_overlap(obj_oobb.aabb, oobb.aabb):
                new_l.append(oobb)
        self.movable_boxes = new_l

        # Eliminate occupancy grid voxels where the object is
        for voxel in self.occupancy_grid.voxels_from_aabb(scale_aabb(obj_oobb.aabb, 1.1)):
            self.occupancy_grid.set_free(voxel)

    def move_robot(self, q, joints, attachment=None):
        """
        Moves the robot model to a desired configuration. If an object is attached, move it as well.

        Args:
            q (tuple): The desired configuration of the robot.
            joints (list): A list of the joints that correspond to the elements of the configuration.
            attachment (list): A list of an attached object's oobb, its attachment grasp, and the
                                object's code in the environment.
        """
        set_joint_positions(self.robot, joints, q)
        if attachment is not None:
            robot_pose = Pose(
                point=Point(x=q[0], y=q[1]),
                euler=Euler(yaw=q[2]),
            )
            obj_pose = multiply(robot_pose, attachment[1])
            set_pose(attachment[2], obj_pose)

    def update_vision_from_voxels(self, voxels):
        """
        Given a set of voxels, mark them as viewed in the visibility grid.

        Args:
            voxels (set): Voxels to mark as viewed
        """
        for voxel in voxels:
            self.visibility_grid.set_free(voxel)


    def gained_vision_from_conf(self, q):
        """
        Given a configuration, compute the voxels corresponding to the cone of vision.

        Args:
            q (tuple): The robot configuration.
        Returns:
            set: A set of voxels that correspond to the gained vision.
        """
        grid = self.static_vis_grid
        voxels = set()
        pose = Pose(point=Point(x=q[0], y=q[1], z=0), euler=[0, 0, q[2]])
        camera_pose = multiply(pose, self.camera_pose)

        for voxel in grid.voxels_from_aabb(grid.aabb):
            center_world = grid.to_world(grid.center_from_voxel(voxel))
            center_camera = tform_point(invert(camera_pose), center_world)
            pixel = pixel_from_point(CAMERA_MATRIX, center_camera)
            if pixel is not None:
                voxel_pos = grid.pose_from_voxel(voxel)[0]
                dist = distance(voxel_pos, camera_pose[0])

                if dist <= FAR:
                    voxels.add(voxel)
        return voxels

    def visibility_voxels_from_path(self, path, attachment=None):
        """
        Finds the set of voxels that correspond to the swept volume traversed by a path in the
        visibility space.

        Args:
            path (list): The path traversed.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            set: A set of voxels occupied by the path.
        """

        voxels = set()
        vis_points = np.array(self.static_vis_grid.occupied_points)
        vis_voxels = np.array(self.static_vis_grid.occupied_voxel_points)

        for q in path:
            aabb = self.aabb_from_q(q)
            vis_idx = np.all( (aabb.lower<=vis_points) & (vis_points<=aabb.upper), axis=1 )
            voxels.update(list([tuple(vp) for vp in vis_voxels[vis_idx]]))
            # voxel = [x for x in self.env.static_vis_grid.voxels_from_aabb()
            #             if self.env.static_vis_grid.contains(x)]
            if attachment is not None:
                obj_aabb = self.movable_object_oobb_from_q(attachment[0], q, attachment[1]).aabb
                vis_idx = np.all((obj_aabb.lower <= vis_points) & (vis_points <= obj_aabb.upper), axis=1)
                voxels.update(list([tuple(vp) for vp in vis_voxels[vis_idx]]))
        return voxels
        
    def visibility_points_from_path(self, path, attachment=None):
        """
        Finds the set of points that correspond to the swept volume traversed by a path in the
        visibility space.

        Args:
            path (list): The path traversed.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            set: A set of points occupied by the path.
        """

        visibility_points = None
        vis_points = np.array(self.visibility_grid.occupied_points)

        for q in path:
            aabb = self.aabb_from_q(q)
            vis_idx = np.all( (aabb.lower<=vis_points) & (vis_points<=aabb.upper), axis=1 )

            if(visibility_points is None):
                visibility_points = vis_points[vis_idx]
            else:
                visibility_points = np.concatenate([visibility_points, vis_points[vis_idx]] , axis=0)

            if attachment is not None:
                aabb = self.movable_object_oobb_from_q(attachment[0], q, attachment[1]).aabb
                vis_idx = np.all((aabb.lower <= vis_points) & (vis_points <= aabb.upper), axis=1)

                if (visibility_points is None):
                    visibility_points = vis_points[vis_idx]
                else:
                    visibility_points = np.concatenate([visibility_points, vis_points[vis_idx]], axis=0)

        return visibility_points


    def obstruction_from_path(self, path, obstruction, ignore_movable=False, attachment=None):
        """
        Finds the set of voxels from the occupied space that a given path enters in contact with.

        Args:
            path (list): The path traversed.
            obstruction (set): A set of voxels that represent additional occupied space taken from the
                            visibility grid.
            ignore_movable (bool): Whether to check collisions with movable obstacles or not.
            attachment(list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            A set of voxels occupied by the path and the first movable object the path collides with.
        """
        occ_points = np.array(self.occupancy_grid.occupied_points)
        occ_points_from_obs = np.array([list(self.occupancy_grid.center_from_voxel(vox))
                                        for vox in obstruction])
        occupancy_points = None
        movable_coll = None
        
        for q in path:
            # Check for obstruction with the obstacles grid.
            aabb = self.aabb_from_q(q)
            vis_idx = np.all( (aabb.lower <= occ_points) & (occ_points <= aabb.upper), axis=1 )


            if(occupancy_points is None):
                occupancy_points = occ_points[vis_idx]

            else:
                occupancy_points = np.concatenate([occupancy_points, occ_points[vis_idx]] , axis=0)

            if len(obstruction) > 0:
                vis_idx_from_obs = np.all( (aabb.lower <= occ_points_from_obs) &
                                           (occ_points_from_obs <= aabb.upper), axis=1 )
                occupancy_points = np.concatenate([occupancy_points,
                                                   occ_points_from_obs[vis_idx_from_obs]], axis=0)

            if attachment is not None:
                aabb = self.movable_object_oobb_from_q(attachment[0], q, attachment[1]).aabb
                vis_idx = np.all((aabb.lower <= occ_points) & (occ_points <= aabb.upper), axis=1)

                if (occupancy_points is None):
                    occupancy_points = occ_points[vis_idx]
                else:
                    occupancy_points = np.concatenate([occupancy_points, occ_points[vis_idx]], axis=0)

                if len(obstruction) > 0:
                    vis_idx_from_obs = np.all((aabb.lower <= occ_points_from_obs) &
                                              (occ_points_from_obs <= aabb.upper), axis=1)
                    occupancy_points = np.concatenate([occupancy_points,
                                                       occ_points_from_obs[vis_idx_from_obs]], axis=0)

            # Check for collision with movable
            if not ignore_movable and movable_coll is None:
                for movable_box in self.movable_boxes:
                    if aabb_overlap(movable_box.aabb, aabb):
                        movable_coll = movable_box
                        break

        return occupancy_points, movable_coll
        
   

    def remove_movable_object(self, movable_obj):
        """
        Given a movable object, remove it from the record of movable objects.

        Args:
            movable_obj : The movable object oobb to remove.
        """
        new_l = []
        for obj in self.movable_boxes:
            if not (all(np.array(obj.aabb.upper) == np.array(movable_obj.aabb.upper)) and
                    all(np.array(obj.aabb.lower) == np.array(movable_obj.aabb.lower))):
                new_l.append(obj)
        self.movable_boxes = new_l



    def setup_occupancy_grid(self):
        """
        Setups the occupancy grid to detect obstacles.
        """
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = Pose(Point(z=0.01))
        surface_aabb = self.room.aabb
        grid = VoxelGrid(
            resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=RED
        )
        self.occupancy_grid = grid

    def setup_movable_boxes(self):
        """
        Setups the movable objects data structure.
        """
        self.movable_boxes = []

    def set_defaults(self, robot):
        """
        Sets the robot joints to their default configuration.

        Args:
            robot (int): The id of the robot object in the environment.
        """
        joints, values = zip(*[(joint_from_name(robot, k), v) for k, v in DEFAULT_JOINTS.items()])
        set_joint_positions(robot, joints, values)

    def setup_robot(self):
        """
        Setups the robot object for manipulation and visualization in the virtual environment.
        """
        MOVO_URDF = "models/srl/movo_description/movo_robotiq_collision.urdf"
        MOVO_PATH = os.path.abspath(MOVO_URDF)
        robot_body = load_pybullet(MOVO_PATH, fixed_base=True)

        self.set_defaults(robot_body)

        self.camera_pose = get_link_pose(robot_body,
                                         link_from_name(robot_body, "kinect2_rgb_optical_frame"))
        return robot_body

    def plot_grids(self, visibility=False, occupancy=False, movable=False):
        """
        Visualizes the different grids in the simulation based on the specified parameters.

        Args:
            visibility (bool): Whether to show the visibility grid or not.
            occupancy (bool): Whether to show the occupancy grid or not.
            movable (bool): Whether to show the detected movable objects or not
        """
        movable_handles = []
        with LockRenderer():
            p.removeAllUserDebugItems()
            if visibility:
                self.visibility_grid.draw_intervals()
            if occupancy:
                self.occupancy_grid.draw_intervals()
            if movable:
                for movable_box in self.movable_boxes:
                    draw_oobb(movable_box, color=YELLOW)
        return

    def get_robot_vision(self):
        """
        Gets the rgb and depth image of the robot

        Returns:
            The pose of the camera and the captured image.
        """

        # 13 is the link of the optical frame of the rgb camera
        camera_link = link_from_name(self.robot, "kinect2_rgb_optical_frame")
        camera_pose = get_link_pose(self.robot, camera_link)

        camera_image = get_image_at_pose(camera_pose, CAMERA_MATRIX, far=FAR, segment=True)

        return camera_pose, camera_image

    def create_closed_room(self, length, width, center=[0, 0], wall_height=2, movable_obstacles=[]):
        """
        Creates a default closed room in the simulation environment.

        Args:
            length (float): Length of the room.
            width (float): Width of the room.
            center : The center of the room. Used mainly to move the robot without changing the frame of reference.
            wall_height (float): Height of the walls.
            movable_obstacles (list): A list of movable objects to include in the environment.
        Returns:
            A Room object created in simulation.
        """

        floor = self.create_pillar(width=width, length=length, color=TAN)
        set_pose(floor, Pose(Point(x=center[0], y=center[1])))

        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1,
                 Pose(point=Point(x=center[0], y=center[1] + length / 2 + wall_thickness / 2, z=wall_height / 2)))
        wall_2 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2,
                 Pose(point=Point(x=center[0], y=center[1] - (length / 2 + wall_thickness / 2), z=wall_height / 2)))
        wall_3 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3,
                 Pose(point=Point(y=center[1], x=center[0] + width / 2 + wall_thickness / 2, z=wall_height / 2)))
        wall_4 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4,
                 Pose(point=Point(y=center[1], x=center[0] - (width / 2 + wall_thickness / 2), z=wall_height / 2)))
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0, 0 + GRID_HEIGHT))
        return Room([wall_1, wall_2, wall_3, wall_4], [floor], aabb, movable_obstacles)

    def create_pillar(self, width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
        """
        Creates a pillar which translates to creating a box in the simulation of the specified dimensions.

        Args:
            width (float): The width of the pillar.
            length (float): The length of the pillar.
            height (float): The height of the pillar.
            color : The specified color of the pillar.
        Returns:
            A box object created in simulation.
        """
        return create_box(w=width, l=length, h=height, color=color, **kwargs)

    def get_centered_aabb(self):
        """
        Gets the aabb of the current position of the robot centered around the origin.

        Returns:
            The centered aabb.
        """
        # TODO: Using the base aabb for simplicity. Change later
        centered_aabb, _ = recenter_oobb((get_aabb(self.robot, link=4), Pose()))
        centered_aabb.lower[2] += centered_aabb.upper[2]
        centered_aabb.upper[2] += centered_aabb.upper[2]
        # Uncmmment these lines if you want the aabb to account for rotations.
        # centered_aabb.lower[1] = centered_aabb.lower[0]
        # centered_aabb.upper[1] = centered_aabb.upper[0]
        return centered_aabb


    def get_centered_oobb(self):
        """
        Gets the oobb of the current position of the robot centered around the origin.

        Returns:
            The centered oobb.
        """
        # TODO: Using the base aabb for simplicity. Change later
        aabb = get_aabb(self.robot, link=4)
        centered_aabb, pose = recenter_oobb((aabb, Pose()))
        return OOBB(centered_aabb, pose)


    def aabb_from_q(self, q):
        """
        Gets the aabb of the specified configuration.

        Args:
            q (tuple): Configuration of the robot.
        Returns:
            The aabb at the given configuration.
        """
        aabb = aabb_from_oobb(self.oobb_from_q(q))
        aabb.upper[2] += 0.5
        return aabb


    def oobb_from_q(self, q):
        """
        Gets the oobb of the specified configuration.

        Args:
            q (tuple): Configuration of the robot.
        Returns:
            The oobb at the given configuration.
        """
        oobb = self.centered_oobb
        pose = Pose(point=Point(x=q[0], y=q[1], z=oobb.pose[0][2]), euler=[0,0,q[2]])
        return OOBB(oobb.aabb, pose)


def distance(vex1, vex2):
    """
    Helper function that returns the Euclidean distance between two tuples of size 2.

    Args:
        vex1 (tuple): The first tuple
        vex2 (tuple): The second tuple
    Returns:
        float: The Euclidean distance between both tuples.
    """
    return ((vex1[0] - vex2[0]) ** 2 + (vex1[1] - vex2[1]) ** 2) ** 0.5


def find_min_angle(beg, end):
    """
    Finds the minimum angle between two angles. (Angle between -PI and PI)

    Args:
        beg (float): Starting angle.
        end (float): End angle.
    Returns:
        float: Minimum angle between both angles.
    """
    if beg > np.pi:
        beg = beg - 2 * np.pi
    if end > np.pi:
        end = end - 2 * np.pi

    return ((end - beg) + np.pi) % (2 * np.pi) - np.pi