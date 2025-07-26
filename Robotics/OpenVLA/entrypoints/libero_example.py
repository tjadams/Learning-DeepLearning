import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import imageio

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
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# Init environment
# Note that pixel resolution affects: output video size, probably other things
pixel_resolution = 512
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
env.set_init_state(init_states[init_state_id])

def get_libero_image(obs):
    img = obs["agentview_image"]
    # rotate 180 degrees to match preprocessing from training
    img = img[::-1, ::-1]
    return img

dummy_action = [0.] * 7
replay_images = []
for step in range(100):
    obs, reward, done, info = env.step(dummy_action)
    img = get_libero_image(obs)
    replay_images.append(img)
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