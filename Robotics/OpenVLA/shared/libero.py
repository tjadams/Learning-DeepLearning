from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os

def init_libero(pixel_resolution):
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
    
    (env, first_obs) = _init_env(task_bddl_file, pixel_resolution, task_suite, task_id)
    
    return (task_id, task, task_name, task_description, task_bddl_file, task_suite_name, env, first_obs)

def _init_env(task_bddl_file, pixel_resolution, task_suite, task_id):
    print("Setting up environment...")
    
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
    first_obs = env.set_init_state(init_states[init_state_id])
    
    print("Environment initialized!")
    
    return (env, first_obs)