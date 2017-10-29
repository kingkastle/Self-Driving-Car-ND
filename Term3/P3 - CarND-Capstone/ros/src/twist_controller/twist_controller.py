import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
	def __init__(self, *args, **kwargs):

		# Get System Parameters: (Just a basic implementation of PID is done, so many are not used)
		self.vehicle_mass = rospy.get_param('~vehicle_mass')
		self.fuel_capacity = rospy.get_param('~fuel_capacity')

		self.total_vehicle_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY

		self.brake_deadband = rospy.get_param('~brake_deadband')
		self.wheel_radius = rospy.get_param('~wheel_radius')
		self.decel_limit = rospy.get_param('~decel_limit')
		self.accel_limit = rospy.get_param('~accel_limit')
		self.wheel_base = rospy.get_param('~wheel_base')
		self.steer_ratio = rospy.get_param('~steer_ratio')
		self.max_lat_accel = rospy.get_param('~max_lat_accel')
		self.max_steer_angle = rospy.get_param('~max_steer_angle')

		# Set Low pass paramters:
		self.tau = 0.2
		self.ts = 0.1
		self.lpf = LowPassFilter(self.tau, self.ts)

		# Define PID object:
		self.yaw_controller = YawController(wheel_base=self.wheel_base,
											steer_ratio=self.steer_ratio,
											min_speed=1.0*0.447, # set to this value
											max_lat_accel=self.max_lat_accel,
											max_steer_angle=self.max_steer_angle)

		self.throttle_pid = PID(kp=0.5, ki=0.1, kd=0.2, mn=self.decel_limit, mx=self.accel_limit)
		self.steer_controller = PID(5., 0.05, 1., -self.max_steer_angle, self.max_steer_angle)

	def control(self, target_speed, target_angular_speed, current_velocity, dbw_enabled, dt):


		if not dbw_enabled:
			# reset PID under MANUAL drive
			self.throttle_pid.reset()
			self.steer_controller.reset()

		# calculate speed difference:
		speed_diff = target_speed - current_velocity

		# Use PID to calculate throttle:
		throttle = self.throttle_pid.step(speed_diff, dt)
		throttle = self.lpf.filt(throttle)  # make it smoother

		# Correct PID fluctuation: don't push throttle when you have to break!
		if speed_diff < 0 and throttle > 0:
			throttle *= -1.

		# rules to set break or throttle based on the sign
		if throttle > 0:
			brake = 0.
		else:
			# define brake in terms of torque:
			brake = abs(self.total_vehicle_mass * self.wheel_radius * throttle)

		# Case special: Stop Line:
		if abs(target_speed - current_velocity) < 1. and target_speed < 1.:
			throttle = 0.
			brake = abs(self.total_vehicle_mass * self.wheel_radius * self.decel_limit)

		# PID steer:
		steer = self.steer_controller.step(target_angular_speed, dt)
		steer += self.yaw_controller.get_steering(target_speed, target_angular_speed, current_velocity)

		# rospy.loginfo("Parameters: {0}, {1}, {2}| speed dif: {3}".format(throttle, brake, steer, speed_diff))
		return throttle, brake, steer
