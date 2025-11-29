# CAREFUL, follower arm will move as fast as possible to match the leader arm's initial position. So make the follower and leader arms start in the same position physically before running this script..

# Inspired by https://huggingface.co/docs/lerobot/en/il_robots
# and https://huggingface.co/docs/lerobot/en/getting_started_real_world_robot
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460830061  \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460825831 \
    --teleop.id=my_awesome_leader_arm