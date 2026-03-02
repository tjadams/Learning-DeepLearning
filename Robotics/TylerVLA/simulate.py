import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

# If available, this is the cleanest route:
# robot_descriptions provides local paths to cached MJCF/URDF assets.
# TODO: import this
from robot_descriptions import so_arm_101_mjcf  # may also be so_arm_101_urdf depending on package version

from inference import load_policy, preprocess_image  # from the code we wrote earlier


def get_fake_camera_image(model, data, width=256, height=256):
    """
    Minimal offscreen render to RGB.
    For a real pipeline, you can use a wrist camera model or attach a MuJoCo camera.
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=None)  # or camera="your_cam_name"
    img = renderer.render()  # HWC uint8 RGB
    return img


def main(run_dir="runs/tyler_vla", command="pick up the block"):
    # Load policy
    policy, tokenizer, j_mean, j_std, device = load_policy(run_dir)

    # Load MuJoCo model (MJCF)
    xml_path = so_arm_101_mjcf.MJCF_PATH  # robot_descriptions provides this
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

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
        print(f"  {n:30s} qpos_adr={adr}")

    # --- You must decide which joints your policy predicts ---
    # For example, if your SO-ARM-101 has 6 joints + gripper, pick those here by name.
    # Replace with the actual names printed above.
    controlled_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        # add gripper joint(s) if present
    ]

    name_to_adr = {n: adr for n, adr in zip(joint_names, joint_qposadr)}
    qpos_indices = [name_to_adr[n] for n in controlled_joint_names if n in name_to_adr]
    assert len(qpos_indices) > 0, "Could not find controlled joints. Update controlled_joint_names."

    # Cache text ids
    text_ids = tokenizer.encode(command, max_len=16).unsqueeze(0).to(device)

    # Viewer loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # (Optional) start from a neutral pose if you have one
        mujoco.mj_forward(model, data)

        last = time.time()
        while viewer.is_running():
            now = time.time()
            dt = now - last
            last = now

            # Render RGB from sim as policy input
            rgb = get_fake_camera_image(model, data)  # HWC uint8
            img_t = preprocess_image(rgb, image_size=128).unsqueeze(0).to(device)

            # Policy predicts normalized joint targets
            with torch.no_grad():
                pred_norm = policy(img_t, text_ids).squeeze(0).cpu().numpy()

            q_des = pred_norm * j_std + j_mean  # de-normalize

            # Write target joints into qpos directly (simple visualization mode)
            # If you want physics-consistent motion, use actuators + ctrl instead.
            qpos = data.qpos.copy()
            for k, qi in enumerate(qpos_indices):
                if k < len(q_des):
                    qpos[qi] = q_des[k]
            data.qpos[:] = qpos

            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(0.01)


if __name__ == "__main__":
    main()