# Inspired by https://huggingface.co/docs/lerobot/en/getting_started_real_world_robot
# Camera value from camera-init.sh, ports from find-robot-ports.sh
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460830061 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 5}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460825831 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true