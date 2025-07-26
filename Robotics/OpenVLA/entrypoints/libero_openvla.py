# TODO: use relative imports correctly with `python -m Robotics.OpenVLA.entrypoints.libero_openvla` 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.openvla import OpenVLA
from shared.libero import init_libero
from shared.openvla_inference import run_libero_task_with_openvla_inference

# params
# prompt = "put the milk in the trash"
prompt = "pick up the alphabet soup and place it in the basket"

# Didn't notice a difference in latency between 128, 512
# pixel_resolution = 128
pixel_resolution = 512

num_steps = 10
# num_steps = 200

def main():
    (_task_id, _task, _task_name, task_description, _task_bddl_file,
     task_suite_name, env, first_obs) = init_libero(pixel_resolution)

    model = OpenVLA()

    unnorm_key = task_suite_name

    run_libero_task_with_openvla_inference(
        model, prompt, unnorm_key, env, task_description, first_obs, num_steps)


if __name__ == "__main__":
    main()
