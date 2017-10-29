#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        # Publish values:
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',  SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # Initialize a basic controller:
        self.controller = Controller()

        # Initialize current and target parameters:
        self.goal_linear = None
        self.goal_yaw_rate = None
        self.current_linear = None
        self.dbw_enabled = True
        self.dt = None
        self.prev_time = rospy.rostime.get_time()

        # Subscribe to all the topics needed
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # run script:
        self.loop()

    def twist_cmd_cb(self, msg):
        self.goal_linear = msg.twist.linear.x
        self.goal_yaw_rate = msg.twist.angular.z

    def current_velocity_cb(self, msg):
        self.current_linear = msg.twist.linear.x

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            self.dt = rospy.rostime.get_time() - self.prev_time
            self.prev_time = rospy.rostime.get_time()
            if self.goal_linear is not None and self.current_linear is not None:

                # Get predicted throttle, brake, and steering using `twist_controller`
                throttle, brake, steering = self.controller.control(self.goal_linear,
                                                                    self.goal_yaw_rate,
                                                                    self.current_linear,
                                                                    self.dbw_enabled,
                                                                    self.dt)
                # Publish PID controls just if autonomous mode is ON:
                if self.dbw_enabled:
                    self.publish(throttle, brake, steering)

            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
