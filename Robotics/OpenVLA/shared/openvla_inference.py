
import time

from shared.libero_openvla_utils import *

def run_libero_task_with_openvla_inference(model, prompt, unnorm_key, env, task_description, first_obs, num_steps):
    print("Starting to perform the Libero task!")
    start_time = time.time()
    replay_images = []
    
    env_obs = first_obs
    
    # TODO: is there some way to watch the simulation as it's happening, 
    # rather than waiting for the video to be made?
    for step in range(num_steps):
        # TODO: add a delay here (dummy actions) due to the env
        # initialization causing objects to drop to the floor for a few steps.
        # This can be seen in the output video.
        
        env_libero_img = get_libero_image(env_obs)
        
        model_obs = model.process_libero_observation(env_libero_img, env_obs)
                
        action = model.get_action(model_obs, prompt, unnorm_key)
        
        # TODO: understand more about why these 3 steps are needed
        # Normalize and apply action
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        action = action.tolist()
        
        env_obs, reward, done, info = env.step(action)
        
        print(f"Processed action! Step #: {step}")
        
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        
        replay_images.append(env_libero_img)
        
        if done:
            print(f"Episode completed successfully in {step} steps!")
            break
    
    if not done:
        print(f"Episode failed to complete in {num_steps} steps!")

    env.close()
    
    save_video(replay_images, task_description)