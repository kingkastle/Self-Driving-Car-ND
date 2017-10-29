#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704
MAX_DECEL = 1.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Subscribers for /current_pose and /vase_waypoints:
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.current_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)


        # subscriber for /traffic_waypoint and /obstacle_waypoint:
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb, queue_size=1)

        # Publisher in final_waypoints
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Member variables
        self.sampling_rate = 10.
        self.car_pose = None
        self.car_orientation = None
        self.waypoints = []
        self.closest_waypoint = None
        self.redlight_wp = -1
        self.max_speed = rospy.get_param('waypoint_loader/velocity')*5/18.
        self.vel = 0

        # run!
        self.process_info()

        rospy.spin()

    def pose_cb(self, msg):
        self.car_pose = msg.pose

    def waypoints_cb(self, msg):
        self.waypoints = [waypoint for waypoint in msg.waypoints]
        self.base_waypoints_sub.unregister()

        # we need to do this action just one time
        rospy.loginfo("Unregistered from /base_waypoints topic")

    def get_final_waypoints(self, car_pose, waypoints, red_light_wp):
        planned_lane = Lane()
        self.closest_waypoint = self.get_closest_waypoint(car_pose.position, waypoints)

        # work with a range of waypoints that starts at an index of zero
        for idx in range(self.closest_waypoint, self.closest_waypoint + LOOKAHEAD_WPS):
            wp_idx = idx % len(waypoints)
            planned_lane.waypoints.append(waypoints[wp_idx])

        for idx in range(len(planned_lane.waypoints) -1):
            # if redlight is detected...
            if red_light_wp != -1:
                # if redlight is close...
                if abs(self.closest_waypoint - red_light_wp) < LOOKAHEAD_WPS:
                    self.vel = self.max_speed - ((LOOKAHEAD_WPS - abs(red_light_wp - self.closest_waypoint - idx))*(self.max_speed/LOOKAHEAD_WPS))

                    # slow down immediately if too close to a red light
                    if abs(self.closest_waypoint - red_light_wp) < 10:
                        self.vel = 0.
            else:
                # accelerate
                self.vel = self.max_speed

            # print('Current Waypoint: {}'.format(self.closest_waypoint))

            planned_lane.waypoints[idx].twist.twist.linear.x = self.vel

        return planned_lane

    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message.
        self.redlight_wp = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_closest_waypoint(self, current_position, waypoints):
        closer_dist = None
        closer_idx = None
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for idx in range(0, len(waypoints)):
            dist = dl(current_position, waypoints[idx].pose.pose.position)
            if closer_dist is None or dist < closer_dist:
                closer_dist = dist
                closer_idx = idx
        return closer_idx

    def process_info(self):
        rate = rospy.Rate(self.sampling_rate)
        while not rospy.is_shutdown():
            if self.waypoints is not None and self.car_pose is not None:
                final_waypoints = self.get_final_waypoints(self.car_pose, self.waypoints, self.redlight_wp)

                # publish final waypoints:
                self.final_waypoints_pub.publish(final_waypoints)
            rate.sleep()


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
