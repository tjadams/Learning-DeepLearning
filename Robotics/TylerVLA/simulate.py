import os
import tempfile
import textwrap
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer
import torch

from robot_descriptions import so_arm101_mj_description

from inference import load_policy, preprocess_image


def _get_simulated_camera_img(model, data, width=256, height=256):
  """
  Minimal offscreen render to RGB.
  For a real pipeline, you can use a wrist camera model or attach a MuJoCo camera.
  """
  renderer = mujoco.Renderer(model, height=height, width=width)
  renderer.update_scene(data, camera=None)  # or camera="your_cam_name"
  img = renderer.render()  # HWC uint8 RGB
  return img


# Scene simulating pick and place set up I have in my apartment
def _build_scene_xml(robot_xml_name: str) -> str:
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

        <!-- Overview camera (top-down) -->
        <camera name="overview_cam" pos="0 0 1.5" euler="0 0 0" fovy="60"/>

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


def init_robot_without_scene():
  xml_path = so_arm101_mj_description.MJCF_PATH  # robot_descriptions provides this
  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)


def _inject_gripper_camera(robot_xml_path: str, out_dir: str) -> str:
  """Parse robot XML, inject a gripper camera, and write to a temp file. Returns temp path."""
  tree = ET.parse(robot_xml_path)
  root = tree.getroot()

  # Find <body name="gripper"> anywhere in the tree
  gripper_body = root.find(".//{*}body[@name='gripper']") or root.find(".//body[@name='gripper']")
  if gripper_body is None:
    raise RuntimeError("Could not find <body name='gripper'> in robot XML. Check body names.")

  cam = ET.SubElement(gripper_body, "camera")
  cam.set("name", "gripper_cam")
  cam.set("pos", "0 0 0.05")
  cam.set("euler", "180 0 0")
  cam.set("fovy", "60")

  with tempfile.NamedTemporaryFile(
      suffix=".xml", mode="wb", delete=False, dir=out_dir
  ) as f:
    tree.write(f, encoding="utf-8", xml_declaration=True)
    tmp_robot_path = f.name

  return tmp_robot_path


def _load_scene_model() -> mujoco.MjModel:
  robot_xml_path = so_arm101_mj_description.MJCF_PATH
  robot_xml_dir = os.path.dirname(robot_xml_path)

  tmp_robot_path = _inject_gripper_camera(robot_xml_path, robot_xml_dir)
  tmp_robot_name = os.path.basename(tmp_robot_path)
  scene_xml = _build_scene_xml(tmp_robot_name)

  # Write scene temp file into the robot's directory so <include file="name.xml"/> resolves correctly.
  with tempfile.NamedTemporaryFile(
      suffix=".xml", mode="w", delete=False, dir=robot_xml_dir
  ) as f:
    f.write(scene_xml)
    tmp_scene_path = f.name

  try:
    model = mujoco.MjModel.from_xml_path(tmp_scene_path)
  finally:
    os.unlink(tmp_scene_path)
    os.unlink(tmp_robot_path)

  return model


def _place_robot_on_table(model: mujoco.MjModel) -> None:
  # Move the robot's base body onto the table surface (z=0.5).
  # <include> merges robot bodies directly into worldbody, so we offset them here.
  SCENE_BODY_NAMES = {"world", "table", "ball", "bowl"}
  TABLE_TOP_Z = 0.5
  ROBOT_X = 0.0
  ROBOT_Y = 0.25   # north edge of table (table spans ±0.3 in Y)
  for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    parent = model.body_parentid[i]
    # Root-level robot bodies: direct children of worldbody (parent==0) not in our scene
    if parent == 0 and (name or "") not in SCENE_BODY_NAMES:
      model.body_pos[i, 0] = ROBOT_X
      model.body_pos[i, 1] = ROBOT_Y
      model.body_pos[i, 2] += TABLE_TOP_Z


def _get_controlled_joint_indices(model: mujoco.MjModel) -> list[int]:
  # Find which qpos indices correspond to arm joints.
  # Hinge joints are in qpos; gripper may be 1-2 joints depending on model.
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

  return qpos_indices


def _run_viewer_loop(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  # run_dir = "runs/tyler_vla"
  # command = "pick up the ball and place it in the bowl"

  # qpos_indices = _get_controlled_joint_indices(model)
  # text_ids = tokenizer.encode(command, max_len=16).unsqueeze(0).to(device)
  # policy, tokenizer, j_mean, j_std, device = load_policy(run_dir)

  print("Launching simulation...")

  gripper_renderer = mujoco.Renderer(model, height=240, width=320)

  try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
      # (Optional) start from a neutral pose if you have one
      mujoco.mj_forward(model, data)

      last = time.time()
      while viewer.is_running():
        now = time.time()
        dt = now - last
        last = now

        # # Render RGB from sim as policy input
        # rgb = _get_simulated_camera_img(model, data)  # HWC uint8
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

        # Render gripper camera (available for policy use; displayed via viewer camera menu)
        gripper_renderer.update_scene(data, camera="gripper_cam")
        gripper_img = gripper_renderer.render()  # HWC uint8 RGB

        time.sleep(0.01)
  finally:
    del gripper_renderer


def run_sim_on_scene():
  model = _load_scene_model()
  _place_robot_on_table(model)

  data = mujoco.MjData(model)
  _run_viewer_loop(model, data)


def main():
  # Simulate robot only
  # init_robot_without_scene()

  # Simulate scene
  run_sim_on_scene()


if __name__ == "__main__":
  main()
