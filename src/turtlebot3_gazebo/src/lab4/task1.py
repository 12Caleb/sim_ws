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

class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cbk, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_pub = self.create_subscription(LaserScan, '/scan', self.laser_cbk, 10)
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier', 10)
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.next_goal_pub = self.create_publisher(PointStamped, 'next_goal',  10)

        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(Odometry, '/odom', self.__ttbot_pose_cbk, 10)
        self.create_subscription(PointStamped, '/clicked_point', self.__clicked_cbk, 10)
        self.target_pub = self.create_publisher(PointStamped, '/target_point', 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.save_client = self.create_client(SaveMap, '/slam_toolbox/save_map')

        self.avoid_limit = 0.28
        self.map, self.map_data = None, None
        self.laser = None
        self.min_dist = np.inf

        self.flags = {'debug': False,
                      'avoiding': False,
                      'new map': False,
                      'first map': False,
                      'recompute': True,
                      'start spin': True,
                      'follow path': True,
                      'continue straight': False,
                      'backup': False,
                      'tune avoid': False,
                      'mapping done': False,
                      }
        self.targeted_frontier = []

        self.path = None
        self.goal_pose = None
        self.frontier = None
        self.ttbot_pose = PoseStamped()
        self.ttbot_pose.pose.position.x = 0.0
        self.ttbot_pose.pose.position.y = 0.0
        self.start_time = 0.0
        self.path_idx = 0
        self.max_speed = 10.0
        self.max_omega = 100.0
        self.cnt = 0
        self.throttle = 0
        self.cmd_speed = 0.0
        self.cmd_heading = 0.0
        self.fail_cnt = 0


        self.forward = 0
        self.avoid_bounds = [10, 80, 10] # +- on forward, angled, sides
        self.front_blocked = False

    def __clicked_cbk(self, data):
        point_x = data.point.x
        point_y = data.point.y

        x1 = self.ttbot_pose.position.x
        x2 = data.point.x
        y1 = self.ttbot_pose.position.y
        y2 = data.point.y
        theta = (np.arctan2(y2-y1, x2-x1))

        self.get_logger().info(f"{(point_y):.3f},{(point_x):.3f} a: {theta:.3f}, {2*np.arccos(self.ttbot_pose.orientation.w)*np.sign(self.ttbot_pose.orientation.z)}")

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose.pose
        '''
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(
                self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y,
                self.ttbot_pose.pose.orientation.z, self.ttbot_pose.pose.orientation.w,
                ))
        '''

    def timer_cb(self):
        #self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function
        if self.flags['mapping done']:
            self.move_ttbot(0.0, 0.0)
            req = SaveMap.Request()
            req.name.data = "map"
            future = self.save_client.call_async(req)
            while not future.done():
                time.sleep(0.1)
            sys.exit()

        if not self.flags['first map']:
            return
        
        if self.flags['start spin']:
            self.cnt += 1
            command = Twist()
            command.angular.z = 0.6
            if self.cnt == 120: # 160
            # if self.spin_cnt == 2:
                self.flags['start spin'] = False
                command.angular.z = 0.0
                self.cnt = 0
            self.cmd_vel_pub.publish(command)
            return
        
        #self.flags['mapping done'] = True

        if self.flags['backup']:
            #self.get_logger().info(f"Counter: {self.cnt}")
            self.cnt += 1
            self.flags['recompute'] = False
            if self.cnt < 10:
                speed = -0.2
                heading = 0
                self.move_ttbot(speed, heading)
                return
            else:
                self.flags['backup'] = False
                self.flags['recompute'] = True
                self.cnt = 0

        if self.flags['recompute']:
            t=True     
            while t:
                self.fail_cnt += 1
                if self.fail_cnt > 10:
                    #self.flags['start spin'] = True
                    self.cnt = 20
                    return
                self.move_ttbot(0.0,0.0)
                self.frontier = self.get_frontier()
                t = False
                target_tuple = self.get_target_frontier()
                # time.sleep(2)
                #self.get_logger().info(f"Target: {target_tuple[0]:.3f}, {target_tuple[1]:.3f}")
                try:
                    self.a_star_path_planner(self.ttbot_pose, target_tuple)
                except KeyError as e:
                    # self.get_logger().info(f"Key Error: {e}, {type(e)}")
                    t = True
                    if self.flags['backup']:
                        self.flags['backup'] = True
                        self.flags['recompute'] = False
                        return
                except ValueError as v:
                    self.get_logger().info(f'mapping done {v}')
                    self.flags['mapping done'] = True
                    return
            self.flags['recompute'] = False
            self.flags['follow path'] = True
            self.flags['orient'] = False
            self.flags['continue straight'] = False
            self.path_idx = 0

        if not self.flags['continue straight']:
            if self.flags['follow path']:
                self.get_path_idx()
                current_goal = self.path.poses[self.path_idx]  
                speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            else:
                speed, heading = 0.0, 0.0
        elif self.flags['orient']:
            current_goal = self.path.poses[self.path_idx]  
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
        else:
            speed, heading = self.max_speed*1.2, 0.0


        if self.flags['avoiding']:
            self.flags['backup'] = True
            if self.flags['continue straight']:
                self.flags['continue straight'] = False
            if self.flags['follow path']:
                self.flags['recompute'] = True
                self.flags['follow path'] = False



        speed, heading = self.avoid(speed, heading)
        #if self.flags['backup']:
            #self.get_logger().info(f's:{speed:.3f}, h:{heading:.3f}')
        self.move_ttbot(speed, heading)
      
        if self.flags['new map']:
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

    def get_target_frontier(self):
        current_position = (self.ttbot_pose.position.x, self.ttbot_pose.position.y)
        
        dist_min = np.inf
        target = (np.inf, np.inf)
        #self.get_logger().info(f'Frontier is {len(self.frontier)} long')
        for i in range(len(self.frontier)):
            r,c = self.frontier[i]
            r,c = self.node_to_map((r, c))
            if (r, c) not in self.targeted_frontier:
                dist = np.sqrt((current_position[0]-r)**2 + (current_position[1]-c)**2)
                if dist < 1.5:
                    dist += 1.7
                if dist < dist_min:
                    dist_min = dist
                    target = (r,c)

        self.targeted_frontier.append(target)

        target_point = PointStamped()  
        target_point.point.x = float(target[0])
        target_point.point.y = float(target[1])
        target_point.point.z = 0.0
        target_point.header.frame_id = 'map'
        target_point.header.stamp.sec = 0
        target_point.header.stamp.nanosec = 0
        self.target_pub.publish(target_point)

        ttbot = PointStamped()
        ttbot.point.x = float(self.ttbot_pose.position.x)
        ttbot.point.y = float(self.ttbot_pose.position.y)
        ttbot.header.frame_id = 'map'
        ttbot.header.stamp = self.get_clock().now().to_msg()
        self.next_goal_pub.publish(ttbot)

        return target

    def map_cbk(self, data):
        self.fail_cnt = 0
        self.map_msg = data
        self.flags['new map'], self.flags['first map'] = True, True
        self.map = np.reshape(data.data, (data.info.height, data.info.width))
        #data = OccupancyGrid()
        self.map_data = {'width': data.info.width, 'height': data.info.height, 'originX': data.info.origin.position.x, 'originY': data.info.origin.position.y, 'resolution': data.info.resolution}

    def data_display(self):
        show_map = self.map.copy()
        show_map[show_map == -1] = 100
        if self.frontier:
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

        marker_array = self.create_marker_array(frontier, scan=False)
        
        t = 1
        if t ==0:
            #points = [(self.ttbot_pose.position.x, self.ttbot_pose.position.y) for _ in range(360)]
            points = [(0,0) for _ in range(360)]
            front_bias = 0.7
            corner_bias = 1
            for i in range(360):
                d = self.avoid_limit
                a = np.deg2rad(i)
                if i < 30:
                    d = d/np.cos(a) * front_bias
                    points[i] = (points[i][0]+d*np.sin(a), points[i][1]+d*np.cos(a))
                elif 30 < i < 60:
                    d *= corner_bias
                    points[i] = (points[i][0]+d*np.sin(a), points[i][1]+d*np.cos(a)) 
                elif 60 < i < 90:
                    d = d/np.sin(a)
                    points[i] = (points[i][0]+d*np.sin(a), points[i][1]+d*np.cos(a))
                elif i > 330:
                    d = d/np.cos(a)*front_bias
                    points[i] = (points[i][0]+d*np.sin(a), points[i][1]+d*np.cos(a))
                elif 330 > i > 300:
                    d *= corner_bias
                    points[i] = (points[i][0]+d*np.sin(a), points[i][1]+d*np.cos(a))
                elif 300 > i > 270:
                    d = -d/np.sin(a)
                    points[i] = (points[i][0]+d*np.sin(a), points[i][1]+d*np.cos(a))
            boundary = self.create_marker_array(points, True)
            for p in points:
                print(f'({p[0]:.2f}, {p[1]:.3f})')


            self.frontier_pub.publish(boundary)
            sys.exit()
        
        self.frontier_pub.publish(marker_array)
        return frontier

    def create_marker_array(self, frontier, scan=False):
        array = MarkerArray()
        id = 0
        for f in frontier:
            x, y = self.node_to_map(f)
            if scan:
                y,x = f

            m = Marker()
            if scan:
                m.header.frame_id='base_scan'
            else:
                m.header.frame_id='map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.type = m.CUBE
            m.id = id
            # m.action = m.ADD
            m.pose.position.z = 0.01
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.color.r = 0.0
            m.color.g = 0.5
            m.color.b = 0.8
            m.color.a = 1.0
            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.01


            id += 1
            array.markers.append(m)
    
        return array

    def convert_node_to_pose(self, node):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"

        x, y = self.node_to_map(node)

        pose.pose.position.y = y
        pose.pose.position.x = x
        
        return pose
    
    def node_to_map(self, node):
        if type(node) == Map_Node:
            node = node.name
            if type(node) == str:
                y,x = node.split(',')
        elif type(node) == tuple and len(node) == 2:
            y, x = node
        else:
            raise(ValueError("Invalide node data type"))
        map_res = self.map_data['resolution']
        map_origin_y = self.map_data['originY']
        map_origin_x = self.map_data['originX']

        x = float((x)) * map_res + map_origin_x
        y = float((y)) * map_res + map_origin_y
        return (x, y)
    
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

    def avoid(self, speed, heading):
        self.flags['avoiding'] = False
        if self.laser == None:
            return 0,0
        # find index (angle) of closest point

        front_bias = 0.8
        corner_bias = 1.2

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

        if min_dist < self.avoid_limit * 1.2 and (45 < min_idx < 90 or 270 < min_idx < 315):
            heading = np.sign(min_idx-180) * 0.2
            speed = max(0.2, 0.1)
            return speed, heading

        if min_dist > self.avoid_limit or 90 < min_idx < 270:
            return(speed, heading)

        #self.get_logger().info(f'in avoid | s:{speed:.3f}, h:{heading:.3f} d: {min_dist:.3f} a: {min_idx} da: {self.laser[min_idx]}')
        speed = min(0, speed) # allow it to back up
        #heading = np.sign(min_idx-180) * 0.2
        self.flags['avoiding'] = True

        self.move_ttbot(0.0,0.0)
        #plt.show()

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

    def a_star_path_planner(self, start_pose, end_pose):
        try:
            self.path = Path()
            self.path.header.frame_id = "map"
        # self.get_logger().info(
        #     'A* planner.\n> start: {},\n> end: {}'.format(start_pose, end_pose))
            self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
            self.path.header.stamp.sec=int(self.start_time)
            self.path.header.stamp.nanosec=0
        except KeyError as e:
            raise KeyError(f'751, {e}')
        # inflate map and create graph

        try:
            if self.fail_cnt == 1 or self.flags['new map']:
                self.mp = MapProcessor(self.map.copy())       # initialize map processor with map files

                kr = self.mp.rect_kernel(8, 1)
                g_kr = self.mp.gaussian_kernel(12, 3) 
                self.mp.inflate_map(kr) 
                #self.get_logger().info(f'{g_kr}')        
                #plt.imshow(self.mp.inf_map_img_array)
                #plt.show()            
                
                self.mp.inflate_map(g_kr,False, True)                     # Inflate boundaries and make binary
                self.mp.get_graph_from_map()
                self.flags['new map'] = False 
                #plt.imshow(self.mp.inf_map_img_array)
                #plt.show()                     # create nodes for each open pixel and connect adjacent nodes with edges and add to mp.map_graph
        except KeyError as e:
            raise KeyError(f'754: {e}')

        if self.convert_to_node(end_pose) not in self.mp.map_graph.g:
            #self.get_logger().info(f'End Pose is in an inflated region: {end_pose}')
            #plt.imshow(mp.inf_map_img_array)
            #plt.show()
            raise KeyError(f'748: {end_pose}')
        if self.convert_to_node(start_pose) not in self.mp.map_graph.g:
            self.get_logger().info(f'Start Pose is in an inflated region: {start_pose}')
            #plt.imshow(mp.inf_map_img_array)
            #plt.show()
            self.flags['backup'] = True
            raise KeyError("758")
        
        #fig, ax = plt.subplots(dpi=100)
        #plt.imshow(self.mp.inf_map_img_array)
        #plt.show()


        line = 0
        try:
            # set start and end of map graph, compute A* and solve
            self.mp.map_graph.root = self.convert_to_node(start_pose)                # starting point of graph
            line = 1
            self.mp.map_graph.end = self.convert_to_node(end_pose)                   # end of graph
            line = 2
            #self.get_logger().info(f"goal node: {self.convert_to_node(end_pose)}")
            line = 2
            as_maze = AStar(self.mp.map_graph)                                   # initialize astar with map graph
            #self.get_logger().info('Solving A*')
        except KeyError as e:
            raise KeyError(f'788, {line} |:| {e}')
        try:
            as_maze.solve(self.mp.map_graph.g[self.mp.map_graph.root], self.mp.map_graph.g[self.mp.map_graph.end])   # get dist and via lists
        except KeyError as e:
            #print(f"mp.map_graph.g: {mp.map_graph.g}")
            #self.get_logger().info(f'End Pose is in an inflate region: {end_pose}')
            #plt.imshow(mp.inf_map_img_array)
            #plt.show()
            raise KeyError(f'770: {self.mp.map_graph.end}, {e}')

        try:
            # reconstruct A* path
            #self.get_logger().info("Reconstructing path")
            path_as,dist_as = as_maze.reconstruct_path(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])  # Get list of nodes in shortest path
            #self.get_logger().info("Converting to poses")
            self.path.poses = [self.convert_node_to_pose(node) for node in path_as]

            path_arr_as = self.mp.draw_path(path_as)
            #ax.imshow(path_arr_as)
            #plt.xlabel("x")
            #plt.ylabel("y")
            self.move_ttbot(0.0,0.0)
            #plt.show()
            
            '''
            self.get_logger().info(f"Finished a_star_path_planner\nlength={len(self.path.poses)}\n\n (x,y)  Path: ")
            for point in self.path.poses:
                self.get_logger().info(f"({point.pose.position.x:.2f}, {point.pose.position.y:.2f})")
            '''
            self.path_pub.publish(self.path)
        except KeyError as e:
            raise KeyError(f'819, {e}')
        return self.path
    
    def get_path_idx(self):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        # find first dist(node) > 0.2m from current position
        self.prev_idx = self.path_idx
        idx = self.path_idx
        dist = self.calc_pos_dist(self.path.poses[idx], self.ttbot_pose)
        #print(f"dist: {dist}")
        if not self.flags['follow path'] and dist < 0.05:
            #self.flags['continue straight'] = True
            self.get_logger().info("Goal reached")
            return
        while dist <= 0.2:
            dist = self.calc_pos_dist(self.path.poses[idx], self.ttbot_pose)
            idx += 1
            if idx >= len(self.path.poses):
                idx -= 1
                self.flags['follow path'] = True
                self.flags['orient'] = False
                self.flags['recompute'] = True
                #self.flags['continue straight'] = True
                break
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
        if type(self.ttbot_pose) == PoseStamped:
            ttbot_pose = self.ttbot_pose.pose
        else:
            ttbot_pose = self.ttbot_pose

        # heading should point from vehicle posistion to goal position
        # speed should be based on distance

        x1 = vehicle_pose.position.x
        x2 = current_goal_pose.position.x
        y1 = vehicle_pose.position.y
        y2 = current_goal_pose.position.y
        theta = np.arctan2(y2-y1, x2-x1)

        # proportional speed control based on distance and heading, closer = slower, pointing wrong = slower
        k_heading = -1.5
        k_heading_dist = -.07
        k_dist = 1.2



        e_heading = np.sign(ttbot_pose.orientation.z)*2*np.arccos(ttbot_pose.orientation.w) - theta
        if e_heading > 3.1415:
            e_heading -= 6.283
        if e_heading < -3.1415:
            e_heading += 6.283
        dist = self.calc_pos_dist(current_goal_pose, ttbot_pose)

        #if np.abs(e_heading) > 3.1415/2:
        #    dist = -dist

        #speed = min(max(dist * k_dist + e_heading * k_heading_dist, -self.max_speed), self.max_speed)
        speed = min(max(dist * k_dist + e_heading * k_heading_dist, 0), self.max_speed)
        
        if np.abs(e_heading) > 3.1415/5:
            speed = 0

        heading = max(min(e_heading * k_heading, self.max_omega), -self.max_omega) 

        self.throttle += 1
        if self.throttle % 6 == 0:
            #self.get_logger().info('E: {:.4f}, HE {:.4f}, S {:.4f}, H {:.4f}, fo:{}, con{}'.format(
            #    dist, e_heading, speed, heading, self.flags['follow path'], self.flags['continue straight']
            #))
            self.throttle = 0
        
        if self.flags['orient']:
            if np.abs(e_heading) < 0.1:
                self.flags['orient'] = False
            speed = 0.0

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """

        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(heading) # rad/s

        self.cmd_vel_pub.publish(cmd_vel)

    def calc_pos_dist(self, pose, cur_pose):
        if type(pose) == PoseStamped:
            pose = pose.pose
        if type(cur_pose) == PoseStamped:
            cur_pose = cur_pose.pose

        delta_x = (pose.position.x - cur_pose.position.x)
        delta_y = (pose.position.y - cur_pose.position.y)
        return np.sqrt(delta_x ** 2 + delta_y ** 2)
    
    def cmd_cb(self, data):
        #data = Twist()
        if np.abs(data.linear.x) > 0.001 or np.abs(data.angular.z) > 0.001 or self.flags['avoiding']:
            self.cmd_speed = data.linear.x
            self.cmd_heading = data.angular.z


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
