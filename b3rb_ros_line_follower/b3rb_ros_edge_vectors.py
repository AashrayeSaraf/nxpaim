# Copyright 2024 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Float32

import numpy as np
import cv2
import math

# synapse_msgs is a package defined at ~/cognipilot/cranium/src/.
# It provides support for ROS2 messages to be used with CogniPilot.
# EdgeVectors is a message type that stores at most two vectors.
# These vectors represent the shape and magnitude of the road edges.
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10

PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.29 # Bottom portion of image to be analyzed for vectors.
VECTOR_MAGNITUDE_MINIMUM = 2.5


class EdgeVectorsPublisher(Node):
    """Initializes edge vector publisher node with the required publishers and subscriptions."""

    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Subscription for LiDAR data.
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debug purposes).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debug purposes).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for steering command.
        self.publisher_steering_command = self.create_publisher(
            Float32,
            '/steering_command',
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        self.lidar_data = None
        self.previous_error = 0.0
        self.previous_time = self.get_clock().now()

    def lidar_callback(self, msg):
        """Handles incoming LiDAR data."""
        self.lidar_data = msg
        # Process the LiDAR data here if needed

    def publish_debug_image(self, publisher, image):
        """Publishes images for debugging purposes."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        """Calculates vector angle in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Right angle vector.
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def compute_vectors_from_image(self, image, thresh):
        """Analyzes the pre-processed image and creates vectors on the road edges, if they exist."""
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        vectors = []
        for i in range(len(contours)):
            coordinates = contours[i][:, 0, :]

            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
            max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

            min_y_coord = min_y_coords[0]
            max_y_coord = max_y_coords[0]

            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if (magnitude > VECTOR_MAGNITUDE_MINIMUM):
                rover_point = [self.image_width / 2, self.lower_image_height]
                middle_point = (min_y_coord + max_y_coord) / 2
                distance = np.linalg.norm(middle_point - rover_point)

                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = np.max(min_y_coords[:, 0])
                else:
                    max_y_coord[0] = np.max(max_y_coords[:, 0])

                vectors.append([list(min_y_coord), list(max_y_coord)])
                vectors[-1].append(distance)

            cv2.line(image, min_y_coord, max_y_coord, BLUE_COLOR, 2)

        return vectors, image

    def process_image_for_edge_vectors(self, image):
        """Processes the image and creates vectors on the road edges in the image, if they exist."""
        self.image_height, self.image_width, color_count = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image.
        threshold_black = 25
        thresh = cv2.threshold(gray, threshold_black, 255, cv2.THRESH_BINARY_INV)[1]

        thresh = thresh[self.image_height - self.lower_image_height:]
        image = image[self.image_height - self.lower_image_height:]
        vectors, image = self.compute_vectors_from_image(image, thresh)

        vectors = sorted(vectors, key=lambda x: x[2])

        half_width = self.image_width / 2
        vectors_left = [i for i in vectors if ((i[0][0] + i[1][0]) / 2) < half_width]
        vectors_right = [i for i in vectors if ((i[0][0] + i[1][0]) / 2) >= half_width]

        final_vectors = []
        for vectors_inst in [vectors_left, vectors_right]:
            if (len(vectors_inst) > 0):
                cv2.line(image, vectors_inst[0][0], vectors_inst[0][1], GREEN_COLOR, 2)
                vectors_inst[0][0][1] += self.upper_image_height
                vectors_inst[0][1][1] += self.upper_image_height
                final_vectors.append(vectors_inst[0][:2])

        self.publish_debug_image(self.publisher_thresh_image, thresh)
        self.publish_debug_image(self.publisher_vector_image, image)

        return final_vectors

    def camera_image_callback(self, message):
        """Analyzes the image received from /camera/image_raw/compressed to detect road edges."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        vectors = self.process_image_for_edge_vectors(image)

        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0
        if (len(vectors) > 0):
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1
        if (len(vectors) > 1):
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)

        # Adjust steering based on detected vectors.
        self.adjust_steering(vectors)

    def adjust_steering(self, vectors):
        """Adjusts the car steering to remain centered between the two detected road edges."""
        if len(vectors) < 2:
            return

        # Calculate the midpoint of the left and right vectors
        left_vector = vectors[0]
        right_vector = vectors[1]

        left_midpoint = [(left_vector[0][0] + left_vector[1][0]) / 2, (left_vector[0][1] + left_vector[1][1]) / 2]
        right_midpoint = [(right_vector[0][0] + right_vector[1][0]) / 2, (right_vector[0][1] + right_vector[1][1]) / 2]

        # Calculate the center of the lane
        lane_center = [(left_midpoint[0] + right_midpoint[0]) / 2, (left_midpoint[1] + right_midpoint[1]) / 2]

        # Calculate the error (deviation from the center)
        image_center_x = self.image_width / 2
        error = image_center_x - lane_center[0]

        # Calculate the current time
        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9  # convert nanoseconds to seconds

        # PD controller
        Kp = 0.005  # Proportional gain
        Kd = 0.001  # Derivative gain

        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        steering_adjustment = Kp * error + Kd * derivative

        # Update previous error and time
        self.previous_error = error
        self.previous_time = current_time

        # Publish the steering command
        steering_command = Float32()
        steering_command.data = float(steering_adjustment)
        self.publisher_steering_command.publish(steering_command)


def main(args=None):
    """Starts the edge vector publisher node."""
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
