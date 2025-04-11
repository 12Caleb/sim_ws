#!/usr/bin/env python3

import rclpy, heapq, os, yaml

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
import sys

from copy import copy
from rclpy.node import Node
from scipy.signal import convolve2d
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
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

    '''
        pretty sure this turns the map image into a 2D black and white array of viable space and obstacles
    '''
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
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0): # inflation size is less than 2x2
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute) # idk, dont need bc kernal size is 5
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy): # for pixels in bounds of kernel centered at obstacle
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape) # make a new empty map same size as original
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]): # for each pixel
                if self.map.image_array[i][j] == 0: # if pixel is 0 -> wall?
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute) # expand by kernel
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array) # max pixel value - min pixel value in inflated map
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r # normalize values [0-1]
        self.inf_map_img_array = np.flipud(self.inf_map_img_array)

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Map_Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
                    # add a node for each open pixel whose name is its position
        # Connect the nodes through edges (add children)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                # for each newly created node
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

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



class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cbk, 10)
        self.cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_pub = self.create_subscription(LaserScan, '/scan', self.laser_cbk, 10)

        self.avoid_limit = 0.4
        self.map, self.map_data = None, None
        self.laser = None
        self.min_dist = np.inf

        self.flags = {'debug':False, 'avoiding': False, 'new map': False}
        self.targeted_frontier = []


        self.forward = 0
        self.avoid_bounds = [10, 80, 10] # +- on forward, angled, sides
        self.front_blocked = False


    def timer_cb(self):
        #self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function
        speed, heading = self.avoid(0.0, 0.0)
        command = Twist()
        command.linear.x = float(speed)
        command.angular.z = float(heading)
        self.cmd_vel.publish(command)
        if self.flags['new map']:
            self.frontier = self.get_frontier()
            self.flags['new map'] = False 
            if self.flags['debug']:
                self.data_display()


        # 0 start by spinning
        # 1 get frontier and target
        # 2 path plan to target and go
        # 3 when target is reached keep going forward until avoiding
        # 4 repeat 1-3 until size of frontier is below some threshold


        #self.get_logger().info(f"s:{speed:.3f} | h:{heading:.3f}, {self.front_blocked}")


    def laser_cbk(self, data):
        self.laser = data.ranges

    def get_target(self):

        current_position = (0,0) # TODO!!
        
        dist_min = np.inf
        target = (0,0)
        for i in range(len(self.frontier)):
            r,c = self.frontier[i]
            if (r, c) not in self.targeted_frontier:
                dist = np.sqrt((current_position[0]-r)**2 + (current_position[1]-c)**2)
                if dist < dist_min:
                    dist_min = dist
                    target = (r,c)

        self.targeted_frontier.append(target)
        return target


    def map_cbk(self, data):
        self.map_msg = data
        self.flags['new map'] = True
        self.get_logger().info("got new map")
        self.map = np.reshape(data.data, (data.info.height, data.info.width))
        self.map_data = {'width': data.info.width, 'height': data.info.height, 'originX': data.info.origin.position.x, 'originY': data.info.origin.position.y}


    def data_display(self):
        show_map = self.map.copy()
        show_map[show_map == -1] = 200
        for x, y in self.frontier:
            show_map[x, y] = 150

        plt.imshow(show_map)
        plt.show()

    def get_frontier(self):
        rows, cols = self.map.shape
        frontier = []
        open_mask = (self.map == 0)
        unknown_mask = (self.map == -1)
        open_loc = np.argwhere(open_mask)

        for r, c in open_loc:
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols and unknown_mask[nr, nc] == 1:
                    frontier.append((r, c))
                    break

        return frontier

    

    def avoid(self, speed, heading):
        self.flags['avoiding'] = False
        if self.laser == None:
            return 0,0
        # find index (angle) of closest point
        min_dist = np.inf
        min_idx = 0
        for i in range(len(self.laser)):
            if self.laser[i] < min_dist:
                min_dist = self.laser[i]
                min_idx = i
        
        if min_dist > self.avoid_limit or 90 < min_idx < 270:
            return(speed, heading)

        speed = 0
        heading = np.sign(min_idx-180) * 0.2
        self.flags['avoiding'] = True

        return speed, heading
        


    def old_avoid(self, x_vel, z_ang):
        speed = x_vel
        heading = z_ang
        self.front_blocked = False
        if self.laser == None:
            return 0,0
        left_close = False
        front_bounds = (self.forward + self.avoid_bounds[0], self.forward - self.avoid_bounds[0]+360)
        angle_bounds = (front_bounds[0]+self.avoid_bounds[1], front_bounds[1]-self.avoid_bounds[1])
        side_bounds = (angle_bounds[0]+self.avoid_bounds[2] , angle_bounds[1]-self.avoid_bounds[2])

        front = self.laser[0: front_bounds[0]] + self.laser[front_bounds[1]: -1]

        min_dist = 10000

        for m in front: # front
            if m < min_dist:
                min_dist = m
            if m < self.avoid_limit:
                x_vel = 0
                self.front_blocked = True

        for m in self.laser[front_bounds[0] : angle_bounds[0]]: #left front
            if m < min_dist:
                min_dist = m
            if m < self.avoid_limit and z_ang > 0:
                z_ang = -0.15
                x_vel = 0

        for m in self.laser[angle_bounds[1]: front_bounds[1]]: # right front
            if m < min_dist:
                min_dist = m
            if m < self.avoid_limit and z_ang < 0:
                z_ang = 0.15
                x_vel = 0

        for m in self.laser[angle_bounds[0]: side_bounds[0]]: # left
            if m < min_dist:
                min_dist = m
            if m < self.avoid_limit:
                z_ang = -0.15
                left_close = True

        for m in self.laser[side_bounds[1]: angle_bounds[1]]: #right
            if m < min_dist:
                min_dist = m
            if m < self.avoid_limit:
                z_ang = 0.15
                if left_close == True:
                    z_ang = 0
                    x_vel = 0
                    self.front_blocked = True

        if ((min_dist - self.min_dist) > 0.01) and self.front_blocked != True:
            x_vel = speed
            z_ang = heading

        self.min_dist = min_dist
        return x_vel, z_ang

def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
