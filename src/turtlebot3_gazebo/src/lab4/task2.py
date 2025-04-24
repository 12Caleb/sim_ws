#!/usr/bin/env python3

import rclpy, heapq, os, yaml, sys, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import copy
from rclpy.node import Node
from PIL import Image, ImageOps
from slam_toolbox.srv import SaveMap
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist, PointStamped

# Import other python packages that you think necessary

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters

        cwd = os.getcwd()
        map_name = os.path.join(cwd, 'src', 'turtlebot3_gazebo', map_name)
        #map_name = os.path.join('ros_ws','src','task_4', map_name)
        #print(f"File name: {map_name}")

        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        map_name2 = map_df.image[0]

        map_name = os.path.join(cwd, 'src', 'turtlebot3_gazebo', 'maps', map_name2)
        #map_name = os.path.join(cwd, 'src','task_4', 'map', map_name)
        #map_name = os.path.join('ros_ws','src','task_4', 'map', map_name)

        im = Image.open(map_name)
        size = (301, 211) 
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]
        #print(xmin, xmax, ymin, ymax)
        f.close()

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self, map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255
        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        img_array = np.flipud(img_array)
        return img_array
class Map_Node():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name
    
    def __lt__(self, other):
        return self.name < other.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)
class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        #self.g_visual = Graph('G')

    def __call__(self):
        for name,node in self.g.items(): # iterate over name and node pairs
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name # adding new root
        elif(end):
            self.end = node.name # adding new end node

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True
class PriorityQueue():
    def __init__(self):
        self.elements = []

    def empty(self):
        return not self.elements

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]
class AStar():
    def __init__(self, in_tree):
        # in_tree
        self.in_tree = in_tree
        self.frontier = PriorityQueue()
        start = in_tree.g[in_tree.root]
        self.frontier.put(start, 0)

        self.dist = {}
        self.via = {}

        self.dist[start] = 0
        self.via[start] = None

    def __heuristic(self, start, end):
        start = self.strtup(start)
        end = self.strtup(end)
        h = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        return h

    def solve(self, start, end):
        while not self.frontier.empty():
            current = self.frontier.get()
            if current == end:
                break
            # print(current, type(current))

            for i in range(len(current.children)):
                next = current.children[i]
                weight = current.weight[i]
                new_dist = self.dist[current] + weight
                if next not in self.dist or new_dist < self.dist[next]:
                    self.dist[next] = new_dist
                    priority = new_dist + self.__heuristic(next, end)
                    self.frontier.put(next, priority)
                    self.via[next] = current

    def strtup(self, graphnode):
        if type(graphnode) == Map_Node:
            graphnode = graphnode.name
            if type(graphnode) == str:
                graphnode = tuple(map(int, graphnode.split(',')))
        return graphnode

    def reconstruct_path(self,sn,en):
        dist = self.dist[en]
        u = en
        path = [u]
        while u != sn:
            u = self.via[u] # next node is how you got to this one
            path.append(u)
        path.reverse()
        return path,dist
class MapProcessor():
    def __init__(self,name):
        self.map = Map(name) # black and white array of obstacles and open space
        self.inf_map_img_array = np.zeros(self.map.image_array.shape) # zero array the size of the map
        self.map_graph = Tree(name) # idk, make a tree?

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) and
            (value != 0)):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] = max(map_array[i][j], value)

    def inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0): # inflation size is less than 2x2
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute) # idk, dont need bc kernal size is 5
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy): # for pixels in bounds of kernel centered at obstacle
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self, kernel, absolute=True, reinflate = False):
        #plt.imshow(self.map)
        #plt.show()
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        # make a new empty map same size as original
        if reinflate:
            map_in = self.inf_map_img_array.copy()
        else:
            self.inf_map_img_array = np.zeros(self.map.image_array.shape)
            #plt.imshow(self.map.image_array)
            #plt.title("mp.map.image_array")
            #plt.show()
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]): # for each pixel
                if not reinflate:
                    if self.map.image_array[i][j] <= 1: # if pixel is 100 -> wall?
                        self.inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute) # expand by kernel
                else:
                    if map_in[i][j] == 1:
                        self.inflate_obstacle(kernel, self.inf_map_img_array, i, j, absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array) # max pixel value - min pixel value in inflated map
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r # normalize values [0-1]
        #plt.imshow(self.inf_map_img_array)
        #plt.title("inflated")
        #plt.show()
        if not reinflate:
            ...
            #self.inf_map_img_array = np.flipud(self.inf_map_img_array)

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] < 1:
                    node = Map_Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
                    # add a node for each open pixel whose name is its position
        # Connect the nodes through edges (add children)
        st_eg_w = 1
        di_eg_w = np.sqrt(2)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                # for each newly created node
                if self.inf_map_img_array[i][j] < 1:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] < 1:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[st_eg_w+self.inf_map_img_array[i-1][j]])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] < 1:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[st_eg_w+self.inf_map_img_array[i+1][j]])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] < 1:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[st_eg_w+self.inf_map_img_array[i][j-1]])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] < 1:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[st_eg_w+self.inf_map_img_array[i][j+1]])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] < 1:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[di_eg_w+self.inf_map_img_array[i-1][j-1]])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] < 1:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[di_eg_w+self.inf_map_img_array[i-1][j+1]])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] < 1:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[di_eg_w+self.inf_map_img_array[i+1][j-1]])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] < 1:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[di_eg_w+self.inf_map_img_array[i+1][j+1]])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size)) # make a size x size array of ones
        return m

    def circle_kernel(self, size):
      y,x = np.ogrid[-size: size+1, -size: size+1]
      mask = x**2+y**2 <= size**2
      return 1*mask
    
    def diamond_kernel(self,size):
        y,x = np.ogrid[-size: size+1, -size: size+1]
        mask = np.abs(x)+np.abs(y) <= size
        return 1*mask

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.name.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array


class Task2(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self):
        super().__init__('task2_node')
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(PointStamped, '/clicked_point', self.__clicked_cbk, 10)
        self.create_subscription(LaserScan, '/scan', self.__laser_cbk, 10)
        self.create_subscription(Odometry, '/odom', self.__odom_cbk, 10)

        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.next_goal_pub = self.create_publisher(PointStamped, 'next_goal',  10)
        self.target_pub = self.create_publisher(PointStamped, '/target_point', 10)

        self.flags = {
            "recompute": True,
            "follow path": False,
            "path blocked": False,
            "back up": False,
            "start path": False,
        }

        self.map_file_name = os.path.join('maps', 'map')
        self.load_map_props(self.map_file_name)
        self.goal_pose = None
        self.path = None
        self.ttbot_pose = PoseStamped()
        self.ttbot_pose.pose.position.x = 0.0
        self.ttbot_pose.pose.position.y = 0.0
        self.start_time = 0.0
        self.path_idx = 0
        self.odom_pose = copy(self.ttbot_pose.pose)
        self.odom_pose.orientation.z = 0.0
        self.odom_pose.orientation.w = 1.0
        self.laser_samples = np.zeros((30,2))
        self.laser_idx = 0
        self.mp = None
        self.cntr = 0
        self.avoid_limit = 0.24
        self.max_speed = 0.4
        self.max_omega = 1.0

    def timer_cb(self):
        # self.get_logger().info('Task2 node is alive.', throttle_duration_sec=2)

        # Wait for goal
        # Map path
        # follow path until blocked
        # localize and place trash can in map
        # replan
        # repeat
        
        if self.flags['recompute']:
            self.path_idx = 0
            self.move_ttbot(0.0, 0.0)
            self.recompute()
            return
        
        if self.flags['follow path']:
            self.get_path_idx()
            current_goal = self.path.poses[self.path_idx] 
            speed, heading = self.path_follower(self.odom_pose, current_goal)
            speed, heading = self.avoid(speed, heading)
            self.move_ttbot(speed, heading)
        
        if self.flags['back up']:
            self.cntr += 1
            self.move_ttbot(-0.2, 0.0)
            if self.cntr > 11:
                self.move_ttbot(0.0, 0.0)
                self.flags['back up'] = False
                self.cntr = 0
            return

        if self.flags['path blocked']:
            can_pose = self.calculate_can_pose()
            if can_pose:
                self.flags['path blocked'] = False
                self.flags['recompute'] = True
                self.add_can_to_map(can_pose)
                #self.add_can_to_map((can_pose[0], can_pose[1]), 1)
                #self.add_can_to_map(can_pose, 2)
                #self.add_can_to_map(can_pose, 3)
            self.move_ttbot(0.0, 0.0)

    def recompute(self):
        if self.goal_pose == None or self.odom_pose == None:
            return

        try:
            self.a_star_path_planner(self.odom_pose, self.goal_pose)
        except KeyError:
            self.goal_pose == None
            self.get_logger().info("Start or end pose is in an invalid region")
            return
        self.flags['follow path'] = True
        self.flags['start path'] = True
        self.flags['recompute'] = False
    
    def add_can_to_map(self, can_pose, flip = 0):
        pose_x, pose_y = can_pose

        #self.get_logger().info(f"{pose_x//self.map_res}, {pose_y//self.map_res}, {self.map_origin[0]//self.map_res}, {self.map_origin[1]//self.map_res} --- FIGURE IT OUT")

        if flip == 0:
            cany = (pose_x - self.map_origin[0]) // self.map_res
            canx = (pose_y - self.map_origin[1]) // self.map_res
        elif flip == 1:
            cany = (pose_x - self.map_origin[0]) // self.map_res
            canx = (pose_y - self.map_origin[1]) // self.map_res
        elif flip == 2:
            canx = (-pose_x - self.map_origin[0]) // self.map_res
            cany = (-pose_y - self.map_origin[1]) // self.map_res            
        elif flip == 3:
            cany = (-pose_x - self.map_origin[0]) // self.map_res
            canx = (-pose_y - self.map_origin[1]) // self.map_res
        
        cany = int(cany)
        canx = int(canx)

        
        circle = self.mp.circle_kernel(4)

        # self.get_logger().info(f"{circle}")
        self.get_logger().info(f"circle center node:({canx}, {cany})")
        self.mp.inflate_obstacle(circle, self.mp.map.image_array, canx, cany, True)
        plt.imshow(self.mp.map.image_array)
        plt.title("Can added to map.image_array")
        #plt.show()

    def calculate_can_pose(self):
        asign = 1
        min_idx = 0
        min_dist = np.inf
        for i in range(len(self.laser)):
            if self.laser[i] < min_dist:
                min_dist = self.laser[i]
                min_idx = i
        m = min_idx
        if min_idx > 180:
            min_idx -= 360
        self.laser_samples[self.laser_idx, 0] = min_idx
        self.laser_samples[self.laser_idx, 1] = min_dist
        self.laser_idx += 1
        #self.get_logger().info(f"idx: {min_idx}, m: {m}, dist: {min_dist}")

        if self.laser_idx >= self.laser_samples.shape[0]:
            ave = np.mean(self.laser_samples, axis=0)
            #self.get_logger().info(f"{self.laser_samples}\n{ave}")
        else:
            return None
        if ave[0] > 0:
            ave[0] += 5
        else:
            ave[0] -= 5
        ave[1] += 0.2
        angle_min = np.deg2rad(ave[0])
        
        if angle_min > np.pi:
            angle_min -= np.pi*2
        theta = np.sign(self.odom_pose.orientation.z)*2*np.arccos(self.odom_pose.orientation.w) + angle_min
        

        x_can=self.odom_pose.position.x + ave[1] * np.cos(theta)
        y_can=self.odom_pose.position.y + ave[1] * np.sin(theta)
        self.get_logger().info(f"min_idx: {ave[0]:.1f}, th:{theta:.2f}, th-a:{theta-angle_min:.2f}, ({x_can:.2f}, {y_can:.2f})")

        
        can = PointStamped()
        can.header.frame_id = 'map'
        can.header.stamp = self.get_clock().now().to_msg()
        can.point.x = float(x_can)
        can.point.y = float(y_can)
        self.target_pub.publish(can)
        self.laser_idx = 0
        return (x_can, y_can)
    '''
    def face_can(self):
        min_idx = 0
        min_dist = np.inf
        for i in range(len(self.laser)):
            if self.laser[i] < min_dist:
                min_dist = self.laser[i]
                min_idx = i

        theta = np.deg2rad(min_idx)
        e_heading = np.sign(self.odom_pose.orientation.z)*2*np.arccos(self.odom_pose.orientation.w) - theta
        if e_heading > 3.1415:
            e_heading -= 6.283
        if e_heading < -3.1415:
            e_heading += 6.283
        if np.abs(e_heading) > 0.03:
            k_heading = -0.3
            heading = max(min(e_heading * k_heading, self.max_omega), -self.max_omega)
            self.get_logger().info(f"h: {heading}, theta:{theta}, e_h:{e_heading}, {self.odom_pose.orientation.z}, {self.odom_pose.orientation.w}", throttle_duration_sec=0.5)
            self.move_ttbot(0.0, heading)
        else:
            self.move_ttbot(0.0, 0.0)
            self.flags['face can'] = False
        return min_dist
    '''
    def move_ttbot(self, speed, heading):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(heading) # rad/s

        self.cmd_vel_pub.publish(cmd_vel)

    def avoid(self, speed, heading):
        self.flags['path blocked'] = False
        if self.laser == None:
            return 0,0
        # find index (angle) of closest point

        front_bias = 0.9+0.1
        corner_bias = 1.5

        min_dist = np.inf
        min_idx = 0
        for i in range(len(self.laser)):
            d = self.laser[i]
            if i < 30:
                d = d*np.cos(np.deg2rad(i))
                d *= front_bias
            elif 30 < i < 60:
                d *= corner_bias
            elif 60 < i < 90:
                d = d*np.sin(np.deg2rad(i))
            elif i > 330:
                d = d*np.cos(np.deg2rad(i))
                d *= front_bias
            elif 330 > i > 300:
                d *= corner_bias
            elif 300 > i > 270:
                d = -d*np.sin(np.deg2rad(i))

            if d < min_dist:
                min_dist = d
                min_idx = i
        '''
        if min_dist < self.avoid_limit * 1.2 and (45 < min_idx < 90 or 270 < min_idx < 315):
            self.get_logger().info("Veering to avoid\n")
            heading = np.sign(min_idx-180) * 0.2
            speed = 0.2
            return 0.0,0.0
            return speed, heading
        '''
        if min_dist > self.avoid_limit or 90 < min_idx < 270:
            return(speed, heading)

        self.get_logger().info(f'in avoid | s:{speed:.3f}, h:{heading:.3f} d: {min_dist:.3f} a: {min_idx} da: {self.laser[min_idx]}')
        speed = min(0, speed) # allow it to back up
        #heading = np.sign(min_idx-180) * 0.2
        self.flags['path blocked'] = True
        self.flags['follow path'] = False
        self.flags['back up'] = True

        self.move_ttbot(0.0,0.0)
        #plt.show()

        return speed, heading

    def a_star_path_planner(self, start_pose, end_pose):
        self.path = Path()
        self.path.header.frame_id = "map"
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.position, end_pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        self.path.header.stamp.sec=int(self.start_time)
        self.path.header.stamp.nanosec=0
        
        # inflate map and create graph
        if self.mp == None:
            self.mp = MapProcessor(self.map_file_name)
            # self.get_logger().info("\n\n\nNew MP\n\n")
        kr = self.mp.rect_kernel(8, 1)
        g_kr = self.mp.gaussian_kernel(12, 3) 
        #self.get_logger().info(f'{kr}\n{g_kr}')
        self.mp.inflate_map(kr, absolute=True, reinflate=False) 
        self.mp.inflate_map(g_kr,False, True)                     # Inflate boundaries and make binary
        self.mp.get_graph_from_map() 
        fig, ax = plt.subplots(dpi=100)
        plt.imshow(self.mp.inf_map_img_array)
        #plt.show()
        

        # set start and end of map graph, compute A* and solve
        self.mp.map_graph.root = self.convert_to_node(start_pose)                # starting point of graph
        self.mp.map_graph.end = self.convert_to_node(end_pose)                   # end of graph
        
        self.get_logger().info(f"goal node: {self.convert_to_node(end_pose)}")

        as_maze = AStar(self.mp.map_graph)                                   # initialize astar with map graph
        self.get_logger().info('Solving A*')

        try:
            as_maze.solve(self.mp.map_graph.g[self.mp.map_graph.root], self.mp.map_graph.g[self.mp.map_graph.end])   # get dist and via lists
        except KeyError:
            print(f"mp.map_graph.g: {self.mp.map_graph.g}")
            raise KeyError(self.mp.map_graph.end)

        # reconstruct A* path
        self.get_logger().info("Reconstructing path")
        path_as,dist_as = as_maze.reconstruct_path(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])  # Get list of nodes in shortest path
        self.get_logger().info("Converting to poses")
        #self.make_pose_path(path_as, end_pose) # convert list of node tuples to poses
        self.path.poses = [self.convert_node_to_pose(node) for node in path_as]

        path_arr_as = self.mp.draw_path(path_as)
        ax.imshow(path_arr_as)
        plt.title("Path")
        plt.xlabel("x")
        plt.ylabel("y")
        #plt.show()
        
        self.path_pub.publish(self.path)
        return self.path
    '''
    def convert_to_node(self, pose_in):
        if type(pose_in) == tuple:
            pose_x = pose_in[0]
            pose_y = pose_in[1]
        else:
            pose_x = pose_in.position.x
            pose_y = pose_in.position.y
        pose_x = (pose_x - self.map_data['originX']) // self.map_data['resolution']
        pose_y = (pose_y - self.map_data['originY']) // self.map_data['resolution']
        return f"{int(pose_y)},{int(pose_x)}"
    '''
    
    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        if type(vehicle_pose) == PoseStamped:
            vehicle_pose = vehicle_pose.pose
        if type(current_goal_pose) == PoseStamped:
            current_goal_pose = current_goal_pose.pose
        if type(self.odom_pose) == PoseStamped:
            ttbot_pose = self.odom_pose.pose
        else:
            ttbot_pose = self.odom_pose

        # heading should point from vehicle posistion to goal position
        # speed should be based on distance

        x1 = vehicle_pose.position.x
        x2 = current_goal_pose.position.x
        y1 = vehicle_pose.position.y
        y2 = current_goal_pose.position.y
        theta = np.arctan2(y2-y1, x2-x1)

        # proportional speed control based on distance and heading, closer = slower, pointing wrong = slower
        k_heading = -1.1
        k_heading_dist = -0.1
        k_dist = 1.0

        e_heading = np.sign(ttbot_pose.orientation.z)*2*np.arccos(ttbot_pose.orientation.w) - theta
        if e_heading > 3.1415:
            e_heading -= 6.283
        if e_heading < -3.1415:
            e_heading += 6.283
        dist = self.calc_pos_dist(current_goal_pose, ttbot_pose)

        #if np.abs(e_heading) > 3.1415/2:
        #    dist = -dist

        #speed = min(max(dist * k_dist + e_heading * k_heading_dist, -self.max_speed), self.max_speed)
        speed = min(max(dist * k_dist + abs(e_heading) * k_heading_dist, 0), self.max_speed)
        
        if np.abs(e_heading) > 3.1415/6:
            speed = 0.0

        if self.flags['start path'] and np.abs(e_heading) > 0.09:
            speed = 0.0
        else:
            self.flags['start path'] = False

        heading = max(min(e_heading * k_heading, self.max_omega), -self.max_omega) 


        #self.get_logger().info('D: {:.4f}, HE {:.4f}, S {:.4f}, H {:.4f}'.format(
        #    dist, e_heading, speed, heading), throttle_duration_sec = 1)

        return speed, heading

    def get_path_idx(self):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        # find first dist(node) > 0.2m from current position
        self.prev_idx = self.path_idx
        idx = self.path_idx
        dist = self.calc_pos_dist(self.path.poses[idx], self.odom_pose)
        #print(f"dist: {dist}")
        if not self.flags['follow path'] and dist < 0.05:
            #self.flags['continue straight'] = True
            self.get_logger().info("Goal reached")
            return
        while dist <= 0.18:
            dist = self.calc_pos_dist(self.path.poses[idx], self.odom_pose)
            idx += 1
            if idx >= len(self.path.poses):
                idx -= 1
                if dist < 0.1:
                    self.flags['follow path'] = False
                    self.flags['recompute'] = True
                    self.path_idx = 0
                    self.goal_pose = None
                    self.get_logger().info("Goal reached")
                break
        if self.flags['follow path']:
            self.path_idx = idx
        if not self.path_idx == self.prev_idx:
            next_goal = PointStamped()
            next_goal.point.x = self.path.poses[self.path_idx].pose.position.x
            next_goal.point.y = self.path.poses[self.path_idx].pose.position.y
            next_goal.point.z = 0.0
            next_goal.header.frame_id = 'map'
            next_goal.header.stamp.sec = 0
            next_goal.header.stamp.nanosec = 0
            self.next_goal_pub.publish(next_goal)

    def calc_pos_dist(self, pose, cur_pose):
        if type(pose) == PoseStamped:
            pose = pose.pose
        if type(cur_pose) == PoseStamped:
            cur_pose = cur_pose.pose

        delta_x = (pose.position.x - cur_pose.position.x)
        delta_y = (pose.position.y - cur_pose.position.y)
        return np.sqrt(delta_x ** 2 + delta_y ** 2)
    
    def convert_to_node(self, pose_in):
        pose_x = pose_in.position.x
        pose_y = pose_in.position.y

        pose_x = (pose_x - self.map_origin[0]) // self.map_res
        pose_y = (pose_y - self.map_origin[1]) // self.map_res

        return f"{int(pose_y)},{int(pose_x)}"
    
    def load_map_props(self, map_file_name):
        cwd = os.getcwd()
        map_name = os.path.join(cwd, 'src','turtlebot3_gazebo', map_file_name)
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        self.map_res = map_df.resolution[0] #0.05 # meters per pixel (node)
        self.map_origin = map_df.origin[0] #(-5.4, -6.3) # map origin
        f.close()
    def convert_node_to_pose(self, node):
        pose = PoseStamped()
        pose.header.stamp.sec=int(self.start_time)
        pose.header.stamp.nanosec = 0
        pose.header.frame_id = "map"
        if type(node) == Map_Node:
            node = node.name
        pose.pose.position.y = (float(node.split(',')[0]) * self.map_res) + self.map_origin[1]
        pose.pose.position.x = (float(node.split(',')[1]) * self.map_res) + self.map_origin[0]
        
        return pose
    def __goal_pose_cbk(self, data):
        self.goal_pose = data.pose
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.position.x, self.goal_pose.position.y))
    def __clicked_cbk(self, data):
        point_x = data.point.x
        point_y = data.point.y
        #point_x = (point_x - self.map_origin[0]) // self.map_res
        #point_y = (point_y - self.map_origin[1]) // self.map_res

        x1 = self.ttbot_pose.position.x
        x2 = data.point.x
        y1 = self.ttbot_pose.position.y
        y2 = data.point.y
        theta = (np.arctan2(y2-y1, x2-x1))

        self.get_logger().info(f"{(point_y):.3f},{(point_x):.3f} a: {theta:.3f}, {2*np.arccos(self.odom_pose.orientation.w)*np.sign(self.odom_pose.orientation.z)}")
    def __ttbot_pose_cbk(self, data):
        # data = PoseWithCovarianceStamped()
        self.ttbot_pose = data.pose.pose
        self.odom_pose = copy(data.pose.pose)
        
        #self.get_logger().info(
        #    'ttbot_pose: {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(
        #        self.ttbot_pose.position.x, self.ttbot_pose.position.y,
        #        self.ttbot_pose.orientation.z, self.ttbot_pose.orientation.w,
        #        ))
    def __laser_cbk(self, data):
        self.laser = data.ranges
    def __odom_cbk(self, data):
        # data = Odometry()
        self.odom_pose = data.pose.pose


def main(args=None):
    rclpy.init(args=args)

    task2 = Task2()

    try:
        rclpy.spin(task2)
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
