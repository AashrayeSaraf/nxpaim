import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
import math
import numpy as np
from synapse_msgs.msg import EdgeVectors, TrafficStatus
from std_msgs.msg import Bool

QOS_PROFILE_DEFAULT = 10

PI = math.pi

LEFT_TURN = +1.0
RIGHT_TURN = -1.0

LEFT_TURN_OBSTACLE = +0.3
RIGHT_TURN_OBSTACLE = -0.3

TURN_MIN = -0.6
TURN_MAX = 0.6
SPEED_MIN = 0.0
SPEED_MAX = 1.8
SPEED_STOP = 0.0
SPEED_25_PERCENT = SPEED_MAX / 4.0
SPEED_50_PERCENT = SPEED_25_PERCENT * 2
SPEED_75_PERCENT = SPEED_25_PERCENT * 3
SPEED_10_PERCENT = SPEED_MAX / 10

THRESHOLD_OBSTACLE_VERTICAL = 1.0
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25

RAMP_TIMER_DURATION = 5.0
SPEED_INCREMENT = 0.02  # Speed increment for gradual acceleration
SPEED_INCREMENT_INTERVAL = 0.1  # Interval for speed increment in seconds

COLLISION_THRESHOLD = 0.1  # Collision threshold distance in meters
REPEAT_VECTOR_THRESHOLD = 10  # Number of times the same vector components can be received before considering it a collision

REVERSE_TIME = 1.5  # Time to reverse in seconds
REVERSE_SPEED = -SPEED_25_PERCENT  # Speed when reversing

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT
        )

        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT
        )

        self.subscription_traffic = self.create_subscription(
            TrafficStatus,
            '/traffic_status',
            self.traffic_status_callback,
            QOS_PROFILE_DEFAULT
        )

        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT
        )

        self.subscription_stop_signal = self.create_subscription(
            Bool,
            '/stop_signal',
            self.stop_signal_callback,
            QOS_PROFILE_DEFAULT
        )


        self.traffic_status = TrafficStatus()
        self.obstacle_detected = False
        self.obstacle_left = False
        self.obstacle_right = False
        self.ramp_detected = False

        self.pid = PID(kp=0.2, ki=0.2, kd=0.05)
        self.last_time = self.get_clock().now()

        self.ramp_timer = None
        self.on_ramp = False

        self.current_speed = SPEED_MIN
        self.target_speed = SPEED_MIN
        self.last_speed_update_time = self.get_clock().now()

        # Variables for collision detection based on repeat vectors
        self.previous_vector = None
        self.repeat_vector_count = 0
        self.prev_reverse_turn = 'NOTURN'

        # Variables for reverse logic
        self.reversing = False
        self.reverse_start_time = None

        self.stopped = False

    def start_ramp_timer(self):
        self.ramp_timer = self.create_timer(RAMP_TIMER_DURATION, self.stop_ramp_timer)

    def stop_ramp_timer(self):
        if self.ramp_timer:
            self.ramp_timer.cancel()
            self.ramp_timer = None
            self.on_ramp = False
            # self.get_logger().info("Stopped ramp timer")

    def rover_move_manual_mode(self, speed, turn):
        self.get_logger().info(f"Sending command - Speed: {speed}, Turn: {turn}")
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)
        self.get_logger().info(f"Joy message published")

    def update_speed(self, target_speed):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_speed_update_time).nanoseconds / 1e9

        if target_speed > self.current_speed:
            if dt >= SPEED_INCREMENT_INTERVAL:
                self.current_speed = min(self.current_speed + SPEED_INCREMENT, target_speed)
                self.last_speed_update_time = current_time
        else:
            # Instant deceleration
            self.current_speed = target_speed

    def reverse_buggy(self):
        self.get_logger().warn("Collision detected! Reversing...")
        self.reversing = True
        self.reverse_start_time = self.get_clock().now()

        # Determine the turn direction while reversing
        if self.obstacle_left:
            reverse_turn = RIGHT_TURN_OBSTACLE
            self.prev_reverse_turn = 'RIGHT'
            self.get_logger().info("Reversing with a right turn due to obstacle on the left")
        elif self.obstacle_right:
            reverse_turn = LEFT_TURN_OBSTACLE
            self.prev_reverse_turn = 'LEFT'
            self.get_logger().info("Reversing with a left turn due to obstacle on the right")
        else:
            # Alternate turns if no specific obstacle side is detected
            if self.prev_reverse_turn == 'LEFT':
                reverse_turn = RIGHT_TURN_OBSTACLE
                self.prev_reverse_turn = 'RIGHT'
                self.get_logger().info("Reversing with a right turn (alternating)")
            else:
                reverse_turn = LEFT_TURN_OBSTACLE
                self.prev_reverse_turn = 'LEFT'
                self.get_logger().info("Reversing with a left turn (alternating)")

        # Reverse with the determined turn direction
        if not self.stopped:
            self.rover_move_manual_mode(REVERSE_SPEED, reverse_turn)
        else:
            self.rover_move_manual_mode(SPEED_STOP, 0.0)

    def stop_reversing(self):
        self.get_logger().info("Stopping reverse, ready to turn")
        self.reversing = False
        self.rover_move_manual_mode(SPEED_STOP, 0.0)

    def edge_vectors_callback(self, message):
        # self.get_logger().info("Edge vectors callback triggered")

        if self.reversing:
            current_time = self.get_clock().now()
            dt = (current_time - self.reverse_start_time).nanoseconds / 1e9
            if dt >= REVERSE_TIME:
                self.stop_reversing()
            else:
                return  # Continue reversing until time is up

        # Check for repeated vector components
        if self.previous_vector == message.vector_1 and message.vector_count == 1:
            self.repeat_vector_count += 1
            # self.get_logger().info(f"Same vector received {self.repeat_vector_count} times")
        else:
            self.repeat_vector_count = 0

        if self.repeat_vector_count > REPEAT_VECTOR_THRESHOLD:
            self.reverse_buggy()
            return

        self.previous_vector = message.vector_1 if message.vector_count == 1 else None

        turn = 0.0
        vectors = message
        half_width = vectors.image_width / 2

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if self.on_ramp:
            if vectors.vector_count == 0:
                self.target_speed = SPEED_10_PERCENT
                turn = 0.0
                # self.get_logger().info("Continuing straight on ramp")
            else:
                self.stop_ramp_timer()
                # self.get_logger().info("Regained vectors, stopping ramp timer")
                self.on_ramp = False
        elif vectors.vector_count == 0:
            if not self.on_ramp:
                self.start_ramp_timer()
                self.on_ramp = True
                # self.get_logger().info("Lost vectors, starting ramp timer")
        else:
            if vectors.vector_count == 1:
                self.target_speed = SPEED_50_PERCENT
                deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
                turn = deviation / vectors.image_width
            elif vectors.vector_count == 2:
                middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2
                middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2
                middle_x = (middle_x_left + middle_x_right) / 2
                deviation = half_width - middle_x

                turn = self.pid.compute(deviation / half_width, dt)
                turn = max(TURN_MIN, min(TURN_MAX, turn))

                if abs(deviation) > half_width / 2:
                    self.target_speed = SPEED_10_PERCENT
                else:
                    self.target_speed = SPEED_MAX

        if self.ramp_detected:
            self.target_speed = SPEED_10_PERCENT
            turn = 0.0
            # self.get_logger().info("Ramp/bridge detected")

        if self.obstacle_detected:
            self.target_speed = SPEED_10_PERCENT
            if self.obstacle_left:
                turn = RIGHT_TURN_OBSTACLE
                # self.get_logger().info("Turning right due to obstacle on the left")
            elif self.obstacle_right:
                turn = LEFT_TURN_OBSTACLE
                # self.get_logger().info("Turning left due to obstacle on the right")

        if not self.stopped:
            self.get_logger().info("Buggy is running-----------------------------------------")
            self.update_speed(self.target_speed)
            self.rover_move_manual_mode(self.current_speed, turn)
        else:
            self.get_logger().info("Buggy is stopped-----------------------------------------")
            self.update_speed(SPEED_STOP)
            self.rover_move_manual_mode(self.current_speed, turn)
        # self.get_logger().info(f"Calculated turn: {turn}, Target speed: {self.target_speed}")

    def traffic_status_callback(self, message):
        if message is not None:
            self.traffic_status = message

    def lidar_callback(self, scan):
        ranges = np.array(scan.ranges)
        ranges[ranges < scan.range_min] = np.nan
        ranges[ranges > scan.range_max] = np.nan
        min_distance = np.nanmin(ranges)

        if min_distance < COLLISION_THRESHOLD:
            self.obstacle_detected = True
            angle_of_min_distance = np.nanargmin(ranges) * scan.angle_increment

            if angle_of_min_distance < PI / 2:
                self.obstacle_left = True
                self.obstacle_right = False
            else:
                self.obstacle_left = False
                self.obstacle_right = True
        else:
            self.obstacle_detected = False
            self.obstacle_left = False
            self.obstacle_right = False

    def stop_signal_callback(self, message):
        if message.data:
            self.get_logger().info("Stop signal received. Stopping buggy.")
            self.stopped = True
            self.update_speed(SPEED_STOP)
            self.rover_move_manual_mode(SPEED_STOP, 0.0)

def main(args=None):
    rclpy.init(args=args)
    line_follower = LineFollower()

    try:
        rclpy.spin(line_follower)
    except KeyboardInterrupt:
        pass

    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
