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
        map_name = os.path.join(cwd, 'src','task_4', map_name)
        #map_name = os.path.join('ros_ws','src','task_4', map_name)
        #print(f"File name: {map_name}")

        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        map_name = map_df.image[0]

        map_name = os.path.join(cwd, 'src','task_4', 'map', map_name)
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
    def __init__(self,map_in):
        map_in[map_in == -1] = 50
        self.map = map_in # black and white array of obstacles and open space
        self.inf_map_img_array = np.zeros(self.map.shape) # zero array the size of the map
        self.map_graph = Tree("mapper") # idk, make a tree?

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) and
            (value != 0)):
            if absolute:
                map_array[i][j] = value
            else:
                #if map_array[i][j] == 0:
                    #map_array[i][j] += value
                map_array[i][j] = max(map_array[i][j], value)

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0): # inflation size is less than 2x2
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute) # idk, dont need bc kernal size is 5
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy): # for pixels in bounds of kernel centered at obstacle
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel, absolute=True, reinflate = False):
        #plt.imshow(self.map)
        #plt.show()
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        # make a new empty map same size as original
        thresh = 100
        if reinflate:
            thresh = 1
            map_in = self.inf_map_img_array.copy()
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]): # for each pixel
                if not reinflate:
                    if self.map[i][j] == thresh: # if pixel is 100 -> wall?
                        self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute) # expand by kernel
                else:
                    if map_in[i][j] == thresh:
                        self.__inflate_obstacle(kernel, self.inf_map_img_array, i, j, absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array) # max pixel value - min pixel value in inflated map
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r # normalize values [0-1]
        # self.inf_map_img_array = np.flipud(self.inf_map_img_array)

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.inf_map_img_array[i][j] < 1:
                    node = Map_Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
                    # add a node for each open pixel whose name is its position
        # Connect the nodes through edges (add children)
        st_eg_w = 1
        di_eg_w = np.sqrt(2)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                # for each newly created node
                if self.inf_map_img_array[i][j] < 1:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] < 1:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[st_eg_w+self.inf_map_img_array[i-1][j]])
                    if (i < (self.map.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] < 1:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[st_eg_w+self.inf_map_img_array[i+1][j]])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] < 1:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[st_eg_w+self.inf_map_img_array[i][j-1]])
                    if (j < (self.map.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] < 1:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[st_eg_w+self.inf_map_img_array[i][j+1]])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] < 1:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[di_eg_w+self.inf_map_img_array[i-1][j-1]])
                    if ((i > 0) and (j < (self.map.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] < 1:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[di_eg_w+self.inf_map_img_array[i-1][j+1]])
                    if ((i < (self.map.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] < 1:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[di_eg_w+self.inf_map_img_array[i+1][j-1]])
                    if ((i < (self.map.shape[0] - 1)) and (j < (self.map.shape[1] - 1))):
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
        self.create_subscription(LaserScan, '/scan', self.laser_cbk, 10)

        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.next_goal_pub = self.create_publisher(PointStamped, 'next_goal',  10)

        # Fill in the initialization member variables that you need

    def timer_cb(self):
        self.get_logger().info('Task2 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function

    # Define function(s) that complete the (automatic) mapping task
    def __goal_pose_cbk(self, data):
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
    def __clicked_cbk(self, data):
        point_x = data.point.x
        point_y = data.point.y
        #point_x = (point_x - self.map_origin[0]) // self.map_res
        #point_y = (point_y - self.map_origin[1]) // self.map_res

        x1 = self.ttbot_pose.pose.position.x
        x2 = data.point.x
        y1 = self.ttbot_pose.pose.position.y
        y2 = data.point.y
        theta = (np.arctan2(y2-y1, x2-x1))

        self.get_logger().info(f"{(point_y):.3f},{(point_x):.3f} a: {theta:.3f}, {2*np.arccos(self.ttbot_pose.pose.orientation.w)*np.sign(self.ttbot_pose.pose.orientation.z)}")
    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        # record odom right before reset with amcl
        self.get_logger().info('Odom pose: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                self.odom_pose.position.x, self.odom_pose.position.y,
                self.odom_pose.orientation.z, self.odom_pose.orientation.w
            ))

        self.ttbot_pose = data.pose
        self.odom_pose = copy(data.pose.pose)
        
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(
                self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y,
                self.ttbot_pose.pose.orientation.z, self.ttbot_pose.pose.orientation.w,
                ))
    def laser_cbk(self, data):
        self.laser = data.ranges


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
