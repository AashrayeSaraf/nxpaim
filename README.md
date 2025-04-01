# NXP AIM India - Smart Car Challenge

## Overview
This repository contains the code and implementation details for our participation in the **NXP AIM India Smart Car Challenge**. Our project focuses on **AI-driven autonomous navigation**, leveraging **deep learning, LiDAR, and ROS 2** to enable self-driving capabilities.

## Features
- **Traffic Sign Detection** using YOLOv5 trained on a custom dataset.
- **PID-based Line Following** for smooth and accurate navigation.
- **LiDAR-based Obstacle Avoidance** ensuring safe maneuverability.
- **ROS 2 Integration** for seamless sensor fusion and real-time control.
- **Simulation & Hardware Deployment** on **Gazebo** and **NXP AIM MR-B3RB buggy**.

## Setup Instructions
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3
- ROS 2 (Foxy or Humble recommended)
- OpenCV
- PyTorch
- TensorFlow Lite (for optimized edge inference)
- LiDAR and camera drivers

### Installation
```bash
# Clone the repository
git clone git@github.com:NXPHoverGames/NXP_AIM_INDIA_2024.git b3rb_ros_line_follower
cd b3rb_ros_line_follower

# Install dependencies
pip install -r requirements.txt
```

## Running the Project
### Simulation (Gazebo)
```bash
source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower vectors
source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower runner
source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower detect
```

### Running on Hardware
```bash
cd ~/cognipilot/cranium/
colcon build
source ~/cognipilot/cranium/install/setup.bash
ros2 launch b3rb_bringup robot.launch.py
source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower vectors
source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower detect
source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower runner
```

## Video Demonstration
[Watch the project in action](https://youtu.be/IiahHKT5NE4)  

## Contributors
- **Aashraye Saraf** – AI Model Development & ROS 2 Integration
- Aishani Sinha, Animesh Lodhi, Shaurya – Navigation, Hardware Implementation, System Optimization


---
For any queries, feel free to open an issue or contact us!
