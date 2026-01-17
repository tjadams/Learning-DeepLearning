import numpy as np
import robosuite as suite
import time

env = suite.make(env_name="Lift",
                 robots="Jaco",
                 has_renderer=True,
                 has_offscreen_renderer=False,
                 use_camera_obs=False,)

env.reset()

MAX_FR = 25  # max frame rate for running simluation

for i in range(1000):
  action = np.random.randn(*env.action_spec[0].shape) * 0.1
  obs, reward, done, info = env.step(action)
  env.render()

  start = time.time()
  # limit frame rate if necessary
  elapsed = time.time() - start
  diff = 1 / MAX_FR - elapsed
  if diff > 0:
    time.sleep(diff)
