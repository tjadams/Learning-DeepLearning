# Inspired by https://huggingface.co/docs/lerobot/en/so101?assembly=Leader#configure-the-motors
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460830061 
    #\ 
    #--robot.id=my_awesome_follower_arm

# Calibration saved to /Users/tjadams/.cache/huggingface/lerobot/calibration/robots/so101_follower/None.json

# Note that I manually copied:
# 1. A backup of None.json to a file in this folder called so-101-follower-calibration-backup.json 
# 2. A backup of None.json to a file in the output folder called my_awesome_follower_arm.json. This is because I think lerobot-teleoperate needs it in teleoperate.sh

# Note that the id associated with a robot is used to store the calibration file. 
# It's important to use the same id when teleoperating, recording, and evaluating when using the same setup.