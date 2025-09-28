# Overview
This section of the repo covers my usage of my SO-ARM-101 and HuggingFace's LeRobot libraries.

# Usage
1. `cd ~/Coding/Learning-DeepLearning/Robotics/SO-ARM-101/`
1. `conda activate lerobot`

# First time usage
1. See usage above
1. `chmod 777 scripts/calibrate-follower.sh`
1. `chmod 777 scripts/calibrate-leader.sh`
1. `./scripts/calibrate-follower.sh` and follow its steps
1. `./scripts/calibrate-leader.sh` and follow its steps

# Notes
For my personal setup, here are the USB ports associated with each arm:
- Leader: /dev/tty.usbmodem5A460825831
- Follower (gripper): /dev/tty.usbmodem5A460830061

The speed at which data is transmitted on the bus for each robot arm, is determined by the baudrate. Each motor is identified by a unique id on the bus. For the communication to work properly between the motors and the controller, we need unique ids for each motor.