from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
import numpy as np
import math


class OpenVLA:
  def __init__(self):
    model_name = "openvla/openvla-7b-finetuned-libero-object"

    # Apple Silicon
    self.device = "mps"

    # torch.bfloat16 not supported on mps
    self.torch_dtype = torch.float16

    self.model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=self.torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(self.device)

    self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print("OpenVLA model initialized!")

  def get_action(self, observation, prompt, unnorm_key):
    image = Image.fromarray(observation["full_image"])
    image = image.convert("RGB")

    prompt = f"In: What action should the robot take to {prompt.lower()}?\nOut:"

    # Process inputs with AutoProcessor
    inputs = self.processor(prompt, image).to(self.device, dtype=self.torch_dtype)

    action = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action

  def process_libero_observation(self, libero_img, obs):
    """
    Process raw libero observation into model input format.
    """

    # Prepare observations dict with state information
    observation = {
        "full_image": libero_img,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                self.quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"]
            )
        ),
    }

    return observation

  def quat2axisangle(self, quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
      quat[3] = 1.0
    elif quat[3] < -1.0:
      quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
      # This is (close to) a zero degree rotation, immediately return
      return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
