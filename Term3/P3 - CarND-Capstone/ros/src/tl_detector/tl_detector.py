#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np


STATE_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub_waypoints = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.car_pose = None
        self.waypoints = None
        self.wp_array = None
        self.camera_image = None
        self.lights = []
        self.min_distance = 50  # min distance to look for traffic signals

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.car_pose = msg.pose

    def waypoints_cb(self, msg):
        self.waypoints = [waypoint for waypoint in msg.waypoints]
        self.wp_array = np.asarray([(w.pose.pose.position.x, w.pose.pose.position.y) for w in self.waypoints])
        self.sub_waypoints.unregister()

        # we need to do this action just one time
        #rospy.loginfo("Unregistered from /base_waypoints topic")

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, car_pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        car_position = np.asarray([self.car_pose.position.x, self.car_pose.position.y])
        dist_squared = np.sum((self.wp_array - car_position)**2, axis=1)
        closer_idx = np.argmin(dist_squared)

        return closer_idx

    def get_closest_waypoint_to_light(self, light_position):
        """Identifies the closest path waypoint to the given light position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            light_position (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        dist_squared = np.sum((self.wp_array - light_position)**2, axis=1)
        closer_idx = np.argmin(dist_squared)

        return closer_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Here the image is transformed to an OpenCV image Object that could be saved.
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if self.car_pose is not None and self.waypoints is not None:

            car_position = np.asarray([self.car_pose.position.x, self.car_pose.position.y])
            min_dist = np.sum(np.sqrt((stop_line_positions - car_position)**2), axis=1)
            closest_stoplight_idx = np.argmin(min_dist)

            # Get the associated waypoint for the closest stop light:
            closest_light = stop_line_positions[closest_stoplight_idx]
            light_wp = self.get_closest_waypoint_to_light(closest_light)

            # Check if the way point is on front of the car at a reasonable distance:
            car_x = self.car_pose.position.x
            light_wp_x = self.waypoints[light_wp].pose.pose.position.x
            if car_x < light_wp_x and (light_wp_x - car_x) < self.min_distance:
                light = self.lights[closest_stoplight_idx]

                # get the state using the classifier:
                state = self.get_light_state(light)

                # comment when testing in real car
                real_state = light.state
                # if real_state != state:
                #     rospy.loginfo("Predicted light state at (%s) meters ahead is (%s), real state is (%s), predicted state is (%s)", min_dist, state, real_state, state)

                return light_wp, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
