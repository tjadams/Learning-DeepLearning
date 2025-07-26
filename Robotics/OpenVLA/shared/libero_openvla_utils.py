import numpy as np
import os
import imageio

def get_libero_image(obs):
    img = obs["agentview_image"]
    # rotate 180 degrees to match preprocessing from training
    img = img[::-1, ::-1]
    return img


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def save_video(replay_images, task_description):
    print(f"Starting to save video...")
    video_dir = "outputs/videos"
    os.makedirs(video_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{video_dir}/episode={0}--prompt={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in replay_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved replay video to {mp4_path}")