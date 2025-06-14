import os
import time
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import imageio
import numpy as np

from shared.openvla import OpenVLA

# def init_libero():
print("Setting up Libero...")

benchmark_dict = benchmark.get_benchmark_dict()
# Note that you need to download the right datasets via
# python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
# where DATASET is chosen from [libero_spatial, libero_object, libero_100, libero_goal, ...
# e.g. python benchmark_scripts/download_libero_datasets.py --datasets libero_object --use-huggingface
task_suite_name = "libero_object"
task_suite = benchmark_dict[task_suite_name]()

# Get a task from Libero
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

print("Libero initialized!")

print(f"Retrieving task {task_id} from suite {task_suite_name}, the " + \
    f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

print("Setting up environment...")

### params
# prompt = "put the milk in the trash"
prompt = "pick up the alphabet soup and place it in the basket"
unnorm_key = task_suite_name

# Didn't notice a difference in latency between 128, 512
# pixel_resolution = 128
pixel_resolution = 512
###

# Init environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": pixel_resolution,
    "camera_widths": pixel_resolution
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
# For benchmarking, Libero fixes the set of initial states
init_states = task_suite.get_task_init_states(task_id)
init_state_id = 0
obs = env.set_init_state(init_states[init_state_id])

def get_libero_image(obs):
    img = obs["agentview_image"]
    # rotate 180 degrees to match preprocessing from training
    img = img[::-1, ::-1]
    return img

print("Environment initialized!")

model = OpenVLA()

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

print("Starting to perform the Libero task!")
start_time = time.time()
replay_images = []
# TODO: is there some way to watch the simulation as it's happening, 
# rather than waiting for the video to be made?
for step in range(200):
    # TODO: add a delay here (dummy actions) due to the env
    # initialization causing objects to drop to the floor for a few seconds
    
    libero_img = get_libero_image(obs)
    
    observation, libero_img = model.process_libero_observation(libero_img, obs)
            
    action = model.get_action(observation, prompt, unnorm_key)
    
    # TODO: understand more about why these 3 steps are needed
    # Normalize and apply action
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    action = action.tolist()
    
    obs, reward, done, info = env.step(action)
    
    print(f"Processed action! Step #: {step}")
    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    replay_images.append(libero_img)
    
    if done:
        print(f"Episode completed successfully in {step} steps!")
        break
env.close()

video_dir = "outputs/videos"
os.makedirs(video_dir, exist_ok=True)
processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
mp4_path = f"{video_dir}/episode={0}--prompt={processed_task_description}.mp4"
video_writer = imageio.get_writer(mp4_path, fps=30)
for img in replay_images:
    video_writer.append_data(img)
video_writer.close()
print(f"Saved replay video to {mp4_path}")


# def main():
    # init_libero()

# if __name__ == "__main__":
    # main()