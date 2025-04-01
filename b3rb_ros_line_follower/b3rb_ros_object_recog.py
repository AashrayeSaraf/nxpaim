import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image as PILImage
import cv2
from sensor_msgs.msg import Joy
QOS_PROFILE_DEFAULT = 10
SPEED_STOP = 0.0

class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')
        # Publisher for the stop signal
        self.publisher_stop = self.create_publisher(Bool, '/stop_signal', 10)

        # Subscription to the camera image topic
        self.subscription_image = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Initialize CV Bridge to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Load a pre-trained Faster R-CNN model from torchvision
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.eval()

        # Transformation to apply to each frame
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Define the class name for a stop sign in COCO dataset (ID 13)
        self.stop_sign_class_id = 13

        self.get_logger().info("Object detection node initialized with Faster R-CNN")

    def image_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert the OpenCV image to a PIL image for processing
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        img_tensor = self.transform(pil_image).unsqueeze(0)

        # Perform object detection
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Extract the prediction results
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        

        # Check if a stop sign is detected
        for label, score in zip(labels, scores):
            if label.item() == self.stop_sign_class_id and score.item() > 0.2:  # Adjust the confidence threshold as needed
                self.get_logger().info(f"Stop sign detected with confidence {score.item():.2f}")
                self.send_stop_signal()
                break

    def send_stop_signal(self):
        # Publish a stop signal to stop the buggy
        stop_msg = Bool()
        stop_msg.data = True
        self.publisher_stop.publish(stop_msg)
        self.get_logger().info("Stop signal sent")

def main(args=None):
    rclpy.init(args=args)
    object_detection = ObjectDetection()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(object_detection)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        object_detection.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
