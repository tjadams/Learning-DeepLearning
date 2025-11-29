# Usage for dataset.repo_id: my HF_USER is the same name as my mac username so it causes some error. So I just use a different folder name. However, I had to change the folder name for each recording due to another error.
# Recordings are stored locally in ~/.cache/huggingface/lerobot/<dataset.repo_id>

# Inspired by https://huggingface.co/docs/lerobot/en/getting_started_real_world_robot
# Camera value from camera-init.sh, ports from find-robot-ports.sh
# Tried to create a HF Dataset at https://huggingface.co/datasets/tjadams/so-arm-101 but got errors
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460830061 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 5}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460825831 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=abc2/so-arm-101 \
    --dataset.num_episodes=1 \
    --dataset.single_task="Grab the blue stress ball"

#--dataset.repo_id=${HF_USER}/so-arm-101 \