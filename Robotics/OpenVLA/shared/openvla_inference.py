
import time

from shared.libero_openvla_utils import *

def run_libero_task_with_openvla_inference(model, prompt, unnorm_key, env, task_description, first_obs, num_steps):
    print("Starting to perform the Libero task!")
    start_time = time.time()
    replay_images = []
    
    obs = first_obs
    
    # TODO: is there some way to watch the simulation as it's happening, 
    # rather than waiting for the video to be made?
    for step in range(num_steps):
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
    
    save_video(replay_images, task_description)