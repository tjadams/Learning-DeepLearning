import os
import tempfile
import textwrap
import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

from robot_descriptions import so_arm101_mj_description

from inference import load_policy, preprocess_image


def get_simulated_camera_img(model, data, width=256, height=256):
  """
  Minimal offscreen render to RGB.
  For a real pipeline, you can use a wrist camera model or attach a MuJoCo camera.
  """
  renderer = mujoco.Renderer(model, height=height, width=width)
  renderer.update_scene(data, camera=None)  # or camera="your_cam_name"
  img = renderer.render()  # HWC uint8 RGB
  return img


# Scene simulating pick and place set up I have in my apartment
def build_scene_xml(robot_xml_name: str) -> str:
  return textwrap.dedent(f"""\
    <mujoco model="tabletop_scene">
      <option timestep="0.002" gravity="0 0 -9.81"/>

      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
        <material name="table_mat" rgba=".8 .6 .4 1"/>
        <material name="ball_mat"  rgba="1 .2 .2 1"/>
        <material name="bowl_mat"  rgba=".9 .9 .9 1"/>
      </asset>

      <worldbody>
        <!-- Lighting -->
        <light name="top" pos="0 0 2" dir="0 0 -1" diffuse=".8 .8 .8"/>

        <!-- Floor -->
        <geom name="floor" type="plane" size="2 2 .1" material="grid" condim="3"/>

        <!-- Table: top surface at z=0.5 -->
        <body name="table" pos="0 0 0.25">
          <geom name="table_top" type="box" size="0.4 0.3 0.025"
                pos="0 0 0.225" material="table_mat" condim="3"/>
          <geom name="leg_fl" type="cylinder" size="0.02 0.225" pos=" 0.35  0.25 0" material="table_mat"/>
          <geom name="leg_fr" type="cylinder" size="0.02 0.225" pos=" 0.35 -0.25 0" material="table_mat"/>
          <geom name="leg_bl" type="cylinder" size="0.02 0.225" pos="-0.35  0.25 0" material="table_mat"/>
          <geom name="leg_br" type="cylinder" size="0.02 0.225" pos="-0.35 -0.25 0" material="table_mat"/>
        </body>

        <!-- Ball: sphere resting on table surface (z = 0.5 + radius = 0.525) -->
        <body name="ball" pos="0.1 0.05 0.525">
          <freejoint/>
          <geom name="ball_geom" type="sphere" size="0.025"
                material="ball_mat" mass="0.05" condim="4" solimp=".99 .99 .01" solref=".01 1"/>
        </body>

        <!-- Bowl: box-wall approximation, placed on table surface -->
        <body name="bowl" pos="-0.1 0.0 0.5">
          <freejoint/>
          <geom name="bowl_bottom" type="cylinder" size="0.05 0.005"
                material="bowl_mat" mass="0.1"/>
          <geom name="bowl_wall_f" type="box" size="0.005 0.05 0.02" pos=" 0.05 0 0.02" material="bowl_mat" mass="0.02"/>
          <geom name="bowl_wall_b" type="box" size="0.005 0.05 0.02" pos="-0.05 0 0.02" material="bowl_mat" mass="0.02"/>
          <geom name="bowl_wall_l" type="box" size="0.05 0.005 0.02" pos="0  0.05 0.02" material="bowl_mat" mass="0.02"/>
          <geom name="bowl_wall_r" type="box" size="0.05 0.005 0.02" pos="0 -0.05 0.02" material="bowl_mat" mass="0.02"/>
        </body>
      </worldbody>

      <!-- Robot arm: include by filename only (temp file lives in same dir) -->
      <include file="{robot_xml_name}"/>

    </mujoco>
  """)


def main(run_dir="runs/tyler_vla", command="pick up the block"):
  # Simulate robot only
  # xml_path = so_arm101_mj_description.MJCF_PATH  # robot_descriptions provides this
  # model = mujoco.MjModel.from_xml_path(xml_path)
  # data = mujoco.MjData(model)

  # Simulate scene
  robot_xml_path = so_arm101_mj_description.MJCF_PATH
  robot_xml_dir = os.path.dirname(robot_xml_path)
  robot_xml_name = os.path.basename(robot_xml_path)
  scene_xml = build_scene_xml(robot_xml_name)

  # Write temp file into the robot's directory so <include file="name.xml"/> resolves correctly.
  with tempfile.NamedTemporaryFile(
      suffix=".xml", mode="w", delete=False, dir=robot_xml_dir
  ) as f:
    f.write(scene_xml)
    tmp_path = f.name

  try:
    model = mujoco.MjModel.from_xml_path(tmp_path)
    data = mujoco.MjData(model)
  finally:
    os.unlink(tmp_path)

  # Find which qpos indices correspond to arm joints
  # Commonly: hinge joints are in qpos; gripper may be 1-2 joints depending on model.
  # A robust way is to list joint names and map the ones you control.
  joint_names = []
  joint_qposadr = []
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    joint_names.append(name)
    joint_qposadr.append(model.jnt_qposadr[j])

  print("Joints in model:")
  for n, adr in zip(joint_names, joint_qposadr):
    print(f"  {(n or '<unnamed>'):30s} qpos_adr={adr}")

  # --- You must decide which joints your policy predicts ---
  # For example, if your SO-ARM-101 has 6 joints + gripper, pick those here by name.
  # Replace with the actual names printed above.
  controlled_joint_names = [
      "1", "2", "3", "4", "5", "6"
      # add gripper joint(s) if present
  ]

  name_to_adr = {n: adr for n, adr in zip(joint_names, joint_qposadr)}
  qpos_indices = [name_to_adr[n] for n in controlled_joint_names if n in name_to_adr]
  assert len(qpos_indices) > 0, "Could not find controlled joints. Update controlled_joint_names."

  # text_ids = tokenizer.encode(command, max_len=16).unsqueeze(0).to(device)

  # policy, tokenizer, j_mean, j_std, device = load_policy(run_dir)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    # (Optional) start from a neutral pose if you have one
    mujoco.mj_forward(model, data)

    last = time.time()
    while viewer.is_running():
      now = time.time()
      dt = now - last
      last = now

      # # Render RGB from sim as policy input
      # rgb = get_simulated_camera_img(model, data)  # HWC uint8
      # img_t = preprocess_image(rgb, image_size=128).unsqueeze(0).to(device)

      # # Policy predicts normalized joint targets
      # with torch.no_grad():
      #   pred_norm = policy(img_t, text_ids).squeeze(0).cpu().numpy()

      # q_des = pred_norm * j_std + j_mean  # de-normalize

      # # Write target joints into qpos directly (simple visualization mode)
      # # If you want physics-consistent motion, use actuators + ctrl instead.
      # qpos = data.qpos.copy()
      # for k, qi in enumerate(qpos_indices):
      #   if k < len(q_des):
      #     qpos[qi] = q_des[k]
      # data.qpos[:] = qpos

      mujoco.mj_forward(model, data)
      viewer.sync()

      time.sleep(0.01)


if __name__ == "__main__":
  main()
