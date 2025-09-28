# Inspired by https://huggingface.co/docs/lerobot/en/so101?assembly=Leader#configure-the-motors
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460825831
    #\ 
    #--robot.id=my_awesome_leader_arm

# Calibration saved to /Users/tjadams/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/None.json

# Note that the id associated with a robot is used to store the calibration file. 
# It's important to use the same id when teleoperating, recording, and evaluating when using the same setup.