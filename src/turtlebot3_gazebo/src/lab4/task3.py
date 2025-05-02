#!/usr/bin/env python3

import rclpy,cv2, os, yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from rclpy.node import Node
from cv_bridge import CvBridge
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image, LaserScan
from task2 import Map_Node, MapProcessor, AStar
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, PoseStamped, Pose, PoseWithCovarianceStamped, PointStamped


# Import other python packages that you think necessary


class Task3(Node):
    """
    Environment localization and navigation task.
    You can also inherit from Task 2 node if most of the code is duplicated
    """
    def __init__(self):
        super().__init__('task3_node')
        self.timer = self.create_timer(0.1, self.timer_cb)
        self.create_subscription(Image, '/camera/image_raw', self.video_cbk, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(Odometry, '/odom', self.__odom_cbk, 10)
        self.create_subscription(LaserScan, '/scan', self.__laser_cbk, 10)

        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.next_goal_pub = self.create_publisher(PointStamped, 'next_goal',  10)
        self.target_pub = self.create_publisher(PointStamped, '/target_point', 10)
        self.goal_array_pub = self.create_publisher(MarkerArray, '/frontier', 10)

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
        self.bridge = CvBridge()
        self.a_min = 10
        self.im_w = 0
        self.im_h = 0
        self.im_center = (0,0)
        self.color_found = [False, False, False] # RGB
        self.ball_info = [None, None, None] # RGB
        self.ball_location = [None, None, None]
        self.wait_color = [0,0,0]
        self.set_goal_list()
        self.flags = {
            'check video':True,
            'show masks': True,
            'recompute':True,
            'follow path': False,
            'start path': False,
            "path blocked": False,
            "back up": False,
            "goal reached": False,
            'localize': False,
        }
        self.check_video_counter = 0
        self.mp = None
        self.cntr = 0
        self.avoid_limit = 0.25
        self.max_speed = 0.4
        self.max_omega = 1.0

        self.tune_file = open('distance_mapping.csv', 'w')
        self.tune_file.write(f'Area, X in Frame, Bot x, Bot y, Bot z, Bot w, Ball x, Ball y\n')

    def timer_cb(self):
        #self.get_logger().info('Task3 node is alive.', throttle_duration_sec=2)
        self.check_video_counter += 1
        if self.check_video_counter > 10: # /10 seconds
            self.check_video_counter = 0
            self.flags['check video'] = True
        else:
            self.flags['check video'] = False

        self.locate_ball()
        self.flags['check video'] = True
        if self.flags['localize']:
            self.get_logger().info(f"Localizing", throttle_duration_sec = 1)
            return
        
        
        # Spin, checking
        if self.flags['goal reached']:
            self.get_logger().info("spinning")
            self.cntr += 1
            self.move_ttbot(0.0, 1.5)
            if self.cntr > 30:
                self.move_ttbot(0.0, 0.0)
                self.flags['goal reached'] = False
                self.flags['recompute'] = True
                self.goal_pose = None
        

        if self.flags['recompute']:
            self.path_idx = 0
            self.move_ttbot(0.0, 0.0)
            try:
                self.recompute()
            except Exception:
                raise KeyboardInterrupt
            return
        # Go to point, checking along the way

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
                self.flags['recompute'] = True
                self.cntr = 0
                self.goal_pose = None
            return




        # Place on map

    # def calc_ball_pose(self):
    def locate_ball(self):
        blue_position = (6.0,-0.7)
        frame_border = 0.30
        done = True
        for c in range(len(self.ball_info)):
            if not self.color_found[c]:
                done = False
            if (not self.color_found[c]) and self.ball_info[c] != None:
                if self.wait_color[c] > 0:
                    self.wait_color[c] -= 1
                    self.flags['localize'] = False
                    self.get_logger().info("Waiting for a color", throttle_duration_sec=0.8)
                    return
                self.flags['localize'] = True
                x, y, area = self.ball_info[c]
                # if horizontal position is not near the center of the image, turn toward it
                if x < self.im_w * frame_border:
                    self.move_ttbot(0.0, 0.1)
                    return
                if x > self.im_w * (1-frame_border):
                    self.move_ttbot(0.0, -0.1)
                    return
                self.move_ttbot(0.0, 0.0)
                # at this point, the ball is near the middle of the frame
                self.tune_file.write(f'{area:.2f}, {x}, {self.odom_pose.position.x:.3f}, {self.odom_pose.position.y:.3f}, {self.odom_pose.orientation.z:.3f}, {self.odom_pose.orientation.w:.3f}, {blue_position[0]}, {blue_position[1]}\n')
                # area = 199062/dist^2 -> dist = sqrt(199062/area)
                dist = np.sqrt(199062/area)
                if dist > 5:
                    self.wait_color[c] = 10
                    self.flags['localize'] = False
                    break
                angle_in_frame = 0.000587*(self.im_center[0] - x)
                angle_from_robot = np.sign(self.odom_pose.orientation.z)*2*np.arccos(self.odom_pose.orientation.w) + angle_in_frame

                x_ball=self.odom_pose.position.x + dist * np.cos(angle_from_robot)
                y_ball=self.odom_pose.position.y + dist * np.sin(angle_from_robot)

                ball = PointStamped()
                ball.header.frame_id = 'map'
                ball.header.stamp = self.get_clock().now().to_msg()
                ball.point.x = float(x_ball)
                ball.point.y = float(y_ball)
                self.target_pub.publish(ball)
                self.color_found[c] = True
                self.ball_location[c] = (x_ball, y_ball)
                self.flags['localize'] = False
                self.add_ball_to_map(x_ball, y_ball)
                self.flags['recompute'] = True
        
        if done:
            self.goal_list = None
        

    def recompute(self):
        if self.goal_pose == None:
            self.goal_pose = self.goal_list.pop()

        try:
            self.a_star_path_planner(self.odom_pose, self.goal_pose)
        except KeyError:
            self.goal_pose == None
            self.get_logger().info("Start or end pose is in an invalid region")
            return
        self.flags['follow path'] = True
        self.flags['start path'] = True
        self.flags['recompute'] = False

    def video_cbk(self, data):
        #run = True
        if self.flags['check video']:
            self.im_h = data.height
            self.im_w = data.width
            self.im_center = (self.im_w//2, self.im_h//2)
            cv_im = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            masks = self.find_colors(cv_im, show=self.flags['show masks'])
            
            for c in range(len(self.color_found)):
                mask = masks[c]
                if not self.color_found[c]:
                    self.ball_info[c] = self.find_centroid(mask)

        #ball_info = self.find_centroid(red_mask)
        #if run==True:
        #    if self.ball == True:
        #        #follow
        #        self.follow(ball_info)
        #    else:
        #        # stop or search
        #        self.search()
            cv2.waitKey(1)

    def find_colors(self, im, show=False):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        s = 205
        v=85
        red_mask, green_mask, blue_mask = None, None, None

        if show:
            lower = np.array([0, s,v])
            upper = np.array([360, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)
            masked = cv2.bitwise_and(im, im, mask=mask)
            cv2.imshow("masked", masked)

        if not self.color_found[0]:
            lower_red = np.array([0, s,v])
            upper_red = np.array([10, 255, 255])

            lower_red2 = np.array([170, s,v])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

        if not self.color_found[2]:
            lower_blue = np.array([90, s,v])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        if not self.color_found[1]:
            lower_green = np.array([40, s,v])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

        if show:
            if not self.color_found[0]:
                cv2.imshow("Red", red_mask)
            if not self.color_found[2]:
                cv2.imshow("Blue", blue_mask)
            if not self.color_found[1]:
                cv2.imshow("Green", green_mask)

        return (red_mask, green_mask, blue_mask)

    def set_goal_list(self):
        self.goal_list = []
        goals = [
            (-4.4, -4.0),
#            (-4.4, -1.5),
            (-4.2, 2.1),
            (-2.0, 2.9),
            (1.0, 3.0),
            (1.1, -0.5),
            #(6.9, -0.1),
            (3.2, 2.0),
            (5.6, 3.5),
            (8.8, 3.3),
            (8.4, -0.1),
#            (8.3, -2.1),
            (8.3, -3.7)
        ]
        for goal in goals:
            g = Pose()
            g.position.x = goal[0]
            g.position.y = goal[1]
            self.goal_list.append(g)
        marker_array = self.create_marker_array(self.goal_list)
        self.goal_array_pub.publish(marker_array)
        self.get_logger().info("Goals Set\n\n")

    def find_centroid(self, mask):
        contours, __ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        max_a = 0
        max_i = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_a:
                max_a = area
                max_i = i

        if max_a < self.a_min:
            return None
        
        perimeter = cv2.arcLength(contours[max_i], True)
        circularity = 4*np.pi*max_a/perimeter/perimeter
        if circularity < 0.85:
            self.get_logger().info(f"Not a circle {circularity}")
            return None

        cx = 0
        cy = 0
        m = cv2.moments(contours[max_i], True)
        if m['m00'] != 0:
            cx = m['m10'] / m['m00']
            cy = m['m01'] / m['m00']

        self.get_logger().info(f"A: {max_a}, X: {cx:.2f}, Y: {cy:.2f}", throttle_duration_sec=1)

        #color = (120,255,0)
        #dot = cv2.circle(mask.copy(), (int(cx), int(cy)), radius=5, color=color, thickness=-1)
        #dot = cv2.circle(dot, self.im_center, radius=5, color=color, thickness=-1)
        #cv2.imshow('mask', dot)
        #self.ball = True

        return (cx, cy, max_a)

    def create_marker_array(self, pose_list):
        array = MarkerArray()
        id = 0
        for p in pose_list:
            x, y = p.position.x, p.position.y
            m = Marker()
            m.header.frame_id='map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.type = m.CUBE
            m.id = id
            #m.action = m.ADD
            m.pose.position.z = 0.01
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.color.r = 0.8
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.03

            id += 1
            array.markers.append(m)
        self.get_logger().info("\n\nMarker array created\n\n")
        return array

    def add_ball_to_map(self, x, y):
        pose_x, pose_y = x, y

        #self.get_logger().info(f"{pose_x//self.map_res}, {pose_y//self.map_res}, {self.map_origin[0]//self.map_res}, {self.map_origin[1]//self.map_res} --- FIGURE IT OUT")

        bally = (pose_x - self.map_origin[0]) // self.map_res
        ballx = (pose_y - self.map_origin[1]) // self.map_res        
        bally = int(bally)
        ballx = int(ballx)        
        circle = self.mp.circle_kernel(3)

        # self.get_logger().info(f"{circle}")
        self.get_logger().info(f"circle center node:({ballx}, {bally})")
        self.mp.inflate_obstacle(circle, self.mp.map.image_array, ballx, bally, True)
        plt.imshow(self.mp.map.image_array)
        plt.title("Can added to map.image_array")
        plt.show()

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
                    self.flags['goal reached'] = True
                    self.cntr = 0
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
    def move_ttbot(self, speed, heading):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(heading) # rad/s

        self.cmd_vel_pub.publish(cmd_vel)
    def __goal_pose_cbk(self, data):
        #data = PoseStamped()
        self.get_logger().info(f"(x,y) | ({data.pose.position.x},{data.pose.position.y})")
        ...
    def __ttbot_pose_cbk(self, data):
        # data = PoseWithCovarianceStamped()
        self.ttbot_pose = data.pose.pose
        self.odom_pose = copy(data.pose.pose)
    def __laser_cbk(self, data):
        self.laser = data.ranges
    def __odom_cbk(self, data):
        # data = Odometry()
        self.odom_pose = data.pose.pose        

def main(args=None):
    rclpy.init(args=args)

    task3 = Task3()

    try:
        rclpy.spin(task3)
    except KeyboardInterrupt:
        pass
    finally:
        task3.tune_file.close()
        task3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
